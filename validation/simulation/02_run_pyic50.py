"""Step 2 - fit every simulated dataset with PyIC50.

    python 02_run_pyic50.py [--script PATH] [--n-boot 200] [--workers N] [--limit N]

--script   PyIC50 file to test (default: PyIC50.py at the repository root)
--n-boot   bootstrap resamples per fit (the real pipeline uses 1000; 200 keeps this step to minutes)
--limit    only the first N datasets of each scenario (for a quick trial run)

Each dataset is fitted exactly as the pipeline's first pass fits one block
(analyze_drug_pass1), and the last result row is taken as the final model.
"""
import argparse
import contextlib
import functools
import importlib.util
import io
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE = None


def _init_worker(script, n_boot):
    global _MODULE
    warnings.filterwarnings("ignore")
    spec = importlib.util.spec_from_file_location("pyic50_under_test", script)
    _MODULE = importlib.util.module_from_spec(spec)
    sys.argv = ["x"]
    spec.loader.exec_module(_MODULE)
    _MODULE._bootstrap_ic50 = functools.partial(_MODULE._bootstrap_ic50, n_boot=n_boot)


def _fit(job):
    dataset_id, scenario, values = job
    block = pd.DataFrame(values, columns=range(12), index=range(4))
    out = dict(dataset_id=dataset_id, scenario=scenario, status="NO_RESULT")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            results, _, _, _ = _MODULE.analyze_drug_pass1(block, "Sim", 2, 100.0, f"{dataset_id}.xls", exact_conc=True)
    except Exception as exc:                                   # a crash is a result too
        out["status"] = f"ERROR: {type(exc).__name__}"
        return out
    if not results:
        return out
    final = results[-1]
    out.update(status=final["Fit_Status"], IC50=final["IC50_uM"], par_lo=final["IC50_CI95_lo"],
               par_hi=final["IC50_CI95_hi"], boot_lo=final["IC50_Boot_lo"], boot_hi=final["IC50_Boot_hi"],
               hill=final["HillSlope"], bottom=final["Bottom"], top=final["Top"], r2=final["R2"],
               n_rows=len(results))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default=os.path.join(HERE, "..", "..", "PyIC50.py"))
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    script = os.path.abspath(args.script)

    data = pd.read_csv(os.path.join(HERE, "datasets.csv"))
    jobs = []
    for (dataset_id, scenario), g in data.groupby(["dataset_id", "scenario"], sort=True):
        if args.limit is not None and int(dataset_id.rsplit("_", 1)[1]) >= args.limit:
            continue
        values = g.pivot(index="rep", columns="well", values="resp").to_numpy()
        jobs.append((dataset_id, scenario, values))

    print(f"{len(jobs)} datasets, {args.workers} workers, n_boot={args.n_boot}, script={script}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(script, args.n_boot)) as pool:
        for k, res in enumerate(pool.map(_fit, jobs, chunksize=4), 1):
            results.append(res)
            if k % 50 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)

    out = pd.DataFrame(results)
    out.to_csv(os.path.join(HERE, "pyic50_results.csv"), index=False)
    print(out["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
