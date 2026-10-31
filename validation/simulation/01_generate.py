"""Step 1 - simulate many noisy dose-response datasets with a known true IC50.

    python 01_generate.py [datasets_per_scenario]      (default 150)

Writes datasets.csv in long format (one row per well) and scenarios.csv. Every
tool in the next steps reads these same numbers, rounded to 4 decimals like a
plate reader. All randomness is seeded, so the files are reproducible.

Layout of every dataset: four replicates; wells 1-11 are a 1/3 serial dilution
from 100 uM and well 12 is the zero-drug control. True curve: Top 3.0, Bottom 0.3.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests"))
import synthetic as syn  # noqa: E402

BASE, FACTOR, TOP, BOTTOM = 100.0, 3.0, 3.0, 0.3

# name, true IC50 (uM), true Hill slope, proportional noise (cv), additive noise (SD), gross outlier?
SCENARIOS = [
    ("S1_noise_3pct",       3.0,  -1.2, 0.03, 0.04, False),
    ("S2_noise_6pct",       3.0,  -1.2, 0.06, 0.04, False),
    ("S3_noise_10pct",      3.0,  -1.2, 0.10, 0.04, False),
    ("S4_steep_curve",      0.5,  -2.5, 0.06, 0.04, False),
    ("S5_shallow_curve",   10.0,  -0.8, 0.06, 0.04, False),
    ("S6_one_outlier",      3.0,  -1.2, 0.06, 0.04, True),
    ("S7_constant_noise",   3.0,  -1.2, 0.00, 0.15, False),   # same SD at every dose: no heteroscedasticity
]


def main(n_per_scenario):
    conc = np.append(syn.serial_concentrations(BASE, FACTOR), 0.0)   # 12 wells, last = control
    rows, meta = [], []
    for s_index, (name, ic50, hill, cv, floor, outlier) in enumerate(SCENARIOS):
        for i in range(n_per_scenario):
            rng = np.random.default_rng(s_index * 100_000 + i)
            block = syn.make_block(rng, base_conc=BASE, factor=FACTOR, ic50=ic50, hill=hill,
                                   top=TOP, bottom=BOTTOM, cv=cv, floor=floor)
            hit = ""
            if outlier:   # one replicate in the curve's transition/plateau reads 70% too low
                row, col = int(rng.integers(0, 4)), int(rng.integers(2, 9))
                block = syn.with_outliers(block, {(row, col): 0.3})
                hit = f"rep{row + 1}_well{col + 1}"
            block = block.round(4)
            dataset_id = f"{name}_{i:03d}"
            meta.append(dict(dataset_id=dataset_id, scenario=name, true_IC50=ic50, true_hill=hill, outlier=hit))
            for rep in range(4):
                for well in range(12):
                    rows.append((dataset_id, name, well + 1, rep + 1, conc[well], block.iat[rep, well]))

    long = pd.DataFrame(rows, columns=["dataset_id", "scenario", "well", "rep", "conc", "resp"])
    long.to_csv(os.path.join(HERE, "datasets.csv"), index=False, float_format="%.10g")
    pd.DataFrame(meta).to_csv(os.path.join(HERE, "datasets_meta.csv"), index=False)
    print(f"{len(meta)} datasets ({n_per_scenario} per scenario, {len(SCENARIOS)} scenarios) -> datasets.csv")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
