Simulation study: PyIC50 versus R's drc on many synthetic datasets
==================================================================

Question: across many noisy datasets with a known true IC50, how close is each tool to the
truth, and do its 95% intervals contain the truth about 95% of the time?

Scenarios (150 datasets each; 4 replicates; wells 1-11 = 1/3 dilutions from 100 uM, well 12 = control;
true curve Top 3.0 / Bottom 0.3; noise = proportional part + additive floor unless stated):

  S1  3% noise           IC50 3,   Hill -1.2
  S2  6% noise           IC50 3,   Hill -1.2
  S3  10% noise          IC50 3,   Hill -1.2
  S4  steep curve        IC50 0.5, Hill -2.5
  S5  shallow curve      IC50 10,  Hill -0.8
  S6  S2 + one outlier   one replicate in wells 3-9 reads 70% too low (random position)
  S7  constant noise     same SD (0.15) at every dose: no heteroscedasticity

Run in order, from this folder:

  python 01_generate.py 150                       -> datasets.csv, datasets_meta.csv
  python 02_run_pyic50.py --script <PyIC50 file>  -> pyic50_results.csv   (about 20 min on 8 cores)
  Rscript 03_drc_fit.R                            -> drc_results.csv       (about 2 min)
  python 04_summarize.py                          -> summary.csv, per_dataset.csv and the tables

Notes
-----
- Both tools read exactly the same numbers (datasets.csv, rounded to 4 decimals).
- drc: default settings, i.e. unweighted least squares and Wald intervals.
  PyIC50: weighted least squares with a variance model, last result row of the first-pass fit,
  parametric interval and a bootstrap interval (200 resamples here; the pipeline uses 1000).
- Single blocks are fitted with the first-pass function. The pipeline's second pass (a lower plateau
  borrowed from other blocks of the same cell line) needs several blocks and is not exercised.
- Errors are on the log10 scale: 0.10 = 26% off. Coverage is the share of datasets whose interval
  contains the true IC50 (nominal 0.95).
- The noise in S1-S6 is heteroscedastic (it grows with the signal), which is the situation weighting is
  designed for. S7 is the opposite case, where weighting has nothing to gain.
