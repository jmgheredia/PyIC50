# Step 3 - fit every simulated dataset with the 'drc' package (Ritz et al. 2015).
#
#   Rscript 03_drc_fit.R
#
# Model LL.4 (four-parameter log-logistic) with drc's DEFAULT settings: unweighted ordinary least
# squares on all 48 wells of each dataset, including the zero-drug control (dose 0), and drc's
# default Wald 95% intervals. The IC50 compared is the relative IC50 (the inflection point).
#
# drm() sometimes returns the equivalent mirror-image parametrisation of a decreasing curve
# (negative slope with Lower and Upper swapped). The IC50 is the same quantity in both, so
# these fits are kept; the column 'mirrored' records how often it happens.

suppressPackageStartupMessages(library(drc))

data <- read.csv("datasets.csv")
ids <- unique(data$dataset_id)
rows <- vector("list", length(ids))

for (k in seq_along(ids)) {
  d <- data[data$dataset_id == ids[k], ]
  fit <- tryCatch(
    drm(resp ~ conc, data = d, fct = LL.4(names = c("Slope", "Lower", "Upper", "IC50"))),
    error = function(e) NULL)
  if (is.null(fit)) {
    rows[[k]] <- data.frame(dataset_id = ids[k], converged = FALSE, IC50 = NA, lo = NA, hi = NA,
                            hill = NA, lower = NA, upper = NA, mirrored = NA)
    next
  }
  est <- coef(fit)
  ci <- tryCatch(confint(fit, level = 0.95), error = function(e) NULL)
  rows[[k]] <- data.frame(
    dataset_id = ids[k], converged = TRUE,
    IC50 = unname(est["IC50:(Intercept)"]),
    lo = if (is.null(ci)) NA else unname(ci["IC50:(Intercept)", 1]),
    hi = if (is.null(ci)) NA else unname(ci["IC50:(Intercept)", 2]),
    hill = -unname(est["Slope:(Intercept)"]),
    lower = unname(est["Lower:(Intercept)"]), upper = unname(est["Upper:(Intercept)"]),
    mirrored = unname(est["Slope:(Intercept)"]) < 0)
}

out <- do.call(rbind, rows)
write.csv(out, "drc_results.csv", row.names = FALSE)
cat(nrow(out), "datasets fitted;", sum(out$converged), "converged;",
    sum(out$mirrored, na.rm = TRUE), "returned the mirror-image parametrisation\n")
