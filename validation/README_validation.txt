Validation against GraphPad Prism and R (drc) with synthetic data
=================================================================

Six synthetic datasets with KNOWN true parameters (truth.csv), each as:
  csv/V#.csv     concentration (uM) + 4 replicates, last row = zero-drug control  -> for GraphPad and R
  plates/        the same numbers as plate-reader files for PyIC50 (two datasets per plate),
                 plus Starting_Concentrations.xlsx (well 1 = 100 uM, 1/3 dilutions)

The data are identical in every tool (rounded to 4 decimals, like a plate reader).

  V1  low noise (3%)                      IC50 3.0   Hill -1.2
  V2  moderate noise (6%)                 IC50 3.0   Hill -1.2
  V3  high noise (10%)                    IC50 3.0   Hill -1.2
  V4  steep curve                         IC50 0.5   Hill -2.5
  V5  shallow curve                       IC50 10    Hill -0.8
  V6  moderate noise + 1 gross outlier    IC50 3.0   Hill -1.2   (replicate 2, well 4 reads 70% low)

Steps
-----
1. PyIC50 (automatic): copy plates/* and PyIC50.py to an empty folder, run `python PyIC50.py`,
   and copy Consolidated_IC50_Results.xlsx (and the plots) into pyic50_output/.
2. drc (R): in this folder run `Rscript drc_fit.R`  ->  drc_results.csv
   One fit per dataset with drc's defaults (unweighted least squares).
3. GraphPad Prism (manual), for each csv/V#.csv:
   - New XY data table: X = numbers, Y = 4 replicate values side by side. Paste the CSV
     (keep the last row, X = 0, the zero-drug control).
   - Analyze > Nonlinear regression > Dose-response - Inhibition >
     "[Inhibitor] vs. response -- Variable slope (four parameters)"  (X is the concentration,
     NOT its log, so the X = 0 row is allowed). Default weighting (none).
   - Write down IC50 and its 95% CI in prism_results.csv (use prism_results_template.csv) and note
     which CI method Prism used (asymptotic or profile likelihood): it changes the interval.
   - For V6, fit once as is and, if you want, once more after excluding the outlier by hand.
4. `python compare.py` prints and saves comparison.csv: each tool's IC50 and 95% CI against the truth.

What to look at
---------------
- log10_error: how far each estimate is from the true IC50 (0 = exact; 0.1 = 26% off).
- true_in_CI: whether the interval contains the truth. With only six datasets this is an
  illustration, not a coverage estimate; the coverage study is in simulation/.
- The IC50 compared everywhere is the relative IC50 (the curve's inflection point).
- PyIC50 uses weighted least squares and Prism's default is unweighted, so small differences are expected.
