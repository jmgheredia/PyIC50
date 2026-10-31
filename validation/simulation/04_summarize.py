"""Step 4 - compare PyIC50 and drc against the true IC50, scenario by scenario.

    python 04_summarize.py

Reads datasets_meta.csv, pyic50_results.csv and drc_results.csv; writes per_dataset.csv and
summary.csv and prints the tables. Errors are on the log10 scale (0.10 = 26% off).

  RMSE      root-mean-square error of log10(estimated / true IC50)
  bias      mean of that error (positive = overestimates the IC50)
  coverage  share of datasets whose 95% interval contains the true IC50 (nominal: 0.95)
  width     median width of the 95% interval, log10(upper / lower)
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
meta = pd.read_csv(os.path.join(HERE, "datasets_meta.csv"))
py = pd.read_csv(os.path.join(HERE, "pyic50_results.csv"))
dr = pd.read_csv(os.path.join(HERE, "drc_results.csv"))

df = meta.merge(py, on="dataset_id", how="left").merge(dr, on="dataset_id", how="left", suffixes=("", "_drc"))
df = df.rename(columns={"IC50": "py_IC50", "IC50_drc": "drc_IC50", "lo": "drc_lo", "hi": "drc_hi"})
if "scenario_x" in df.columns:
    df = df.rename(columns={"scenario_x": "scenario"})
df["py_ok"] = (df["status"] == "OK") & df["py_IC50"].gt(0)
df["drc_ok"] = df["converged"].fillna(False).astype(bool) & df["drc_IC50"].gt(0)
df["py_err"] = np.where(df["py_ok"], np.log10(df["py_IC50"] / df["true_IC50"]), np.nan)
df["drc_err"] = np.where(df["drc_ok"], np.log10(df["drc_IC50"] / df["true_IC50"]), np.nan)
df.to_csv(os.path.join(HERE, "per_dataset.csv"), index=False)


def covered(lo, hi, truth):
    return ((lo <= truth) & (truth <= hi)).astype(float).where(lo.notna() & hi.notna())


rows = []
for scenario, g in df.groupby("scenario", sort=True):
    both = g["py_ok"] & g["drc_ok"]
    row = dict(scenario=scenario, n=len(g),
               py_fitted=int(g["py_ok"].sum()), drc_fitted=int(g["drc_ok"].sum()),
               py_RMSE=np.sqrt(np.nanmean(g["py_err"] ** 2)), drc_RMSE=np.sqrt(np.nanmean(g["drc_err"] ** 2)),
               py_bias=np.nanmean(g["py_err"]), drc_bias=np.nanmean(g["drc_err"]),
               py_cov_param=covered(g["par_lo"], g["par_hi"], g["true_IC50"])[g["py_ok"]].mean(),
               py_cov_boot=covered(g["boot_lo"], g["boot_hi"], g["true_IC50"])[g["py_ok"]].mean(),
               drc_cov=covered(g["drc_lo"], g["drc_hi"], g["true_IC50"])[g["drc_ok"]].mean(),
               py_width_param=np.nanmedian(np.log10(g["par_hi"] / g["par_lo"])[g["py_ok"]]),
               py_width_boot=np.nanmedian(np.log10(g["boot_hi"] / g["boot_lo"])[g["py_ok"]]),
               drc_width=np.nanmedian(np.log10(g["drc_hi"] / g["drc_lo"])[g["drc_ok"]]),
               py_closer=float((g["py_err"].abs()[both] < g["drc_err"].abs()[both]).mean()))
    rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(HERE, "summary.csv"), index=False)

pd.set_option("display.width", 220)
fmt = lambda v: f"{v:.3f}"
print("\nPOINT ESTIMATE (log10 scale)")
print(summary[["scenario", "n", "py_fitted", "drc_fitted", "py_RMSE", "drc_RMSE", "py_bias", "drc_bias", "py_closer"]]
      .to_string(index=False, float_format=fmt))
print("\n95% INTERVAL (nominal coverage 0.95)")
print(summary[["scenario", "py_cov_param", "py_cov_boot", "drc_cov", "py_width_param", "py_width_boot", "drc_width"]]
      .to_string(index=False, float_format=fmt))
print("\npy_closer = share of datasets where PyIC50's estimate is nearer the truth than drc's")
