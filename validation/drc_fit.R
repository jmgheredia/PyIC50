# Independent reference fit with the 'drc' package (Ritz et al. 2015, Dose-Response Analysis Using R).
#
# Usage (from the validation folder):
#   Rscript drc_fit.R                 # reads csv/*.csv, writes drc_results.csv
#
# Each CSV has X = concentration (uM) and four replicate columns; the last row is the
# zero-drug control. One fit per dataset with the four-parameter log-logistic model LL.4 and
# drc's default settings (unweighted ordinary least squares, Wald 95% intervals), the same as
# the default of GraphPad Prism's nonlinear regression. The IC50 reported is the relative IC50
# (the inflection point), which is what PyIC50 and Prism's
# "[Inhibitor] vs. response - Variable slope (four parameters)" report.

suppressPackageStartupMessages(library(drc))

args <- commandArgs(trailingOnly = TRUE)
csv_dir <- if (length(args) >= 1) args[1] else "csv"
out_file <- if (length(args) >= 2) args[2] else "drc_results.csv"

results <- list()
for (f in sort(list.files(csv_dir, pattern = "\\.csv$", full.names = TRUE))) {
  dataset <- sub("\\.csv$", "", basename(f))
  wide <- read.csv(f)
  reps <- grep("^Rep", names(wide), value = TRUE)
  long <- data.frame(conc = rep(wide$Concentration_uM, times = length(reps)),
                     resp = unlist(wide[reps], use.names = FALSE))
  model <- tryCatch(
    drm(resp ~ conc, data = long, fct = LL.4(names = c("Slope", "Lower", "Upper", "IC50"))),
    error = function(e) NULL)
  if (is.null(model)) {
    results[[length(results) + 1]] <- data.frame(dataset = dataset, IC50 = NA, IC50_lo = NA, IC50_hi = NA,
                                                 Hill = NA, Lower = NA, Upper = NA, note = "did not converge")
    next
  }
  est <- coef(model)
  ci <- confint(model, level = 0.95)
  results[[length(results) + 1]] <- data.frame(
    dataset = dataset,
    IC50 = unname(est["IC50:(Intercept)"]),
    IC50_lo = unname(ci["IC50:(Intercept)", 1]),
    IC50_hi = unname(ci["IC50:(Intercept)", 2]),
    Hill = -unname(est["Slope:(Intercept)"]),   # drc's slope is positive for a decreasing curve
    Lower = unname(est["Lower:(Intercept)"]),
    Upper = unname(est["Upper:(Intercept)"]),
    note = "")
}

table <- do.call(rbind, results)
write.csv(table, out_file, row.names = FALSE)
print(table, digits = 4)
