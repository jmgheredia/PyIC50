"""Put the IC50 estimates of PyIC50, R's drc and (optionally) GraphPad Prism next to the true value.

Usage (from the validation folder):
    python compare.py

Reads:
    truth.csv                                      the values used to simulate each dataset
    pyic50_output/Consolidated_IC50_Results.xlsx   PyIC50's output for the plates in plates/
    drc_results.csv                                output of drc_fit.R
    prism_results.csv  (optional)                  your GraphPad results: dataset,IC50,IC50_lo,IC50_hi
Writes comparison.csv and prints the same table.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
path = lambda *p: os.path.join(HERE, *p)

truth = pd.read_csv(path("truth.csv")).set_index("dataset")
rows = []

# PyIC50: the last row of a block is its final model; it carries the bootstrap interval
consolidated = pd.read_excel(path("pyic50_output", "Consolidated_IC50_Results.xlsx"))
for dataset, group in consolidated.groupby("Drug", sort=True):
    final = group.iloc[-1]
    if final["Fit_Status"] != "OK":
        rows.append(dict(dataset=dataset, method="PyIC50 (parametric CI)", IC50=np.nan, IC50_lo=np.nan, IC50_hi=np.nan))
        continue
    rows.append(dict(dataset=dataset, method="PyIC50 (parametric CI)", IC50=final["IC50_uM"],
                     IC50_lo=final["IC50_CI95_lo"], IC50_hi=final["IC50_CI95_hi"], Hill=final["HillSlope"]))
    rows.append(dict(dataset=dataset, method="PyIC50 (bootstrap CI)", IC50=final["IC50_uM"],
                     IC50_lo=final["IC50_Boot_lo"], IC50_hi=final["IC50_Boot_hi"], Hill=final["HillSlope"]))

for _, r in pd.read_csv(path("drc_results.csv")).iterrows():
    rows.append(dict(dataset=r["dataset"], method="drc (unweighted)", IC50=r["IC50"],
                     IC50_lo=r["IC50_lo"], IC50_hi=r["IC50_hi"], Hill=r["Hill"]))

if os.path.exists(path("prism_results.csv")):
    for _, r in pd.read_csv(path("prism_results.csv")).iterrows():
        rows.append(dict(dataset=r["dataset"], method="GraphPad Prism", IC50=r["IC50"],
                         IC50_lo=r["IC50_lo"], IC50_hi=r["IC50_hi"]))

table = pd.DataFrame(rows)
table["true_IC50"] = table["dataset"].map(truth["true_IC50_uM"])
table["log10_error"] = np.log10(table["IC50"] / table["true_IC50"])
table["true_in_CI"] = (table["IC50_lo"] <= table["true_IC50"]) & (table["true_IC50"] <= table["IC50_hi"])
table["CI_width_log10"] = np.log10(table["IC50_hi"] / table["IC50_lo"])
table = table.sort_values(["dataset", "method"])
table.to_csv(path("comparison.csv"), index=False)

pd.set_option("display.width", 200)
print(table[["dataset", "method", "true_IC50", "IC50", "IC50_lo", "IC50_hi", "log10_error", "true_in_CI"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
