# PyIC50

Automated, reproducible 4-parameter logistic (4PL) dose-response fitting and IC50 estimation from 96-well plate cytotoxicity and cell-viability assays.

<!-- After the first Zenodo release, add the DOI badge here: [![DOI](https://zenodo.org/badge/DOI/YOUR-DOI.svg)](https://doi.org/YOUR-DOI) -->

**Status:** pre-release (0.x). Developed and run with Python 3.12 on Windows.

## Overview

`PyIC50` batch-processes a folder of raw plate-reader exports (`.xls`). Cell line, drug and well-1 concentration are taken from the file name and a registry spreadsheet, every plate goes through the same fixed set of documented quality-control rules, and a weighted 4PL curve is fitted to each drug/cell-line block. The output is one consolidated Excel table, one plot per fitted curve, and a text note for every block that could not be fitted, explaining why.

Quality-control decisions that are usually made by eye (which wells to exclude, whether a low-signal region is a real plateau, how to treat a saturated well) are implemented as explicit rules, so two people running the same version on the same raw data obtain the same estimates.

## Main features

* **Weighted least squares (WLS) fitting.** Weights follow a variance model (variance = a + b x fitted value squared) estimated from the data, so the noisy high-signal wells count less than the tight low-signal ones, and the covariance is scaled by the residual variance.
* **AIC-optimised leave-one-out (LOO) point removal.** A point is dropped only if the Akaike Information Criterion improves by at least 2, up to a maximum of 12 points, and the loop stops once R² reaches 0.90. A final **reinstatement pass** restores dropped points that sit within 1.5 SD of the converged curve.
* **Replicate outlier handling.** Median-absolute-deviation (MAD) flagging within each concentration column, and 95% prediction-band filtering to detect bimodal replicate failures (for example pipetting drops).
* **Dose-response gate.** Unless anchoring has already established a real drop (see below), a block is fitted only if the highest doses sit at least 15% below the baseline (see "How each block is analysed"). Flat or inverted data get no IC50 and no plot, and a note says so.
* **Plateau anchoring.** Curves that drop only at the highest doses ("late drop"), and blocks with clearly dead replicates, get their lower plateau anchored from local data. A late-drop curve with too few dead wells to anchor from can borrow a lower plateau estimated from other blocks of the same cell line (two-pass fallback).
* **Detector saturation.** Readings marked `(+)` are treated as right-censored: they are imputed with the highest reliable reading in the file, flagged as saturated, and excluded from outlier statistics.
* **Same drug, two stocks.** Two stocks of one drug on the same plate also get a third, combined fit (see below).
* **Diluted re-reads.** If a plate was also read after dilution, saturated wells can be recovered from the diluted read, without assuming a linear relationship (see below).
* **Configurable serial dilution.** The dilution factor (1/3 by default) is read per drug from an optional fourth column of `Starting_Concentrations.xlsx`, so plates with 1/2, 1/3 or 1/4 dilutions run without editing the script.
* **Uncertainty.** Parametric 95% confidence intervals from the fit covariance (t distribution), checked by simulation (see "Validation"), and a bootstrap interval (1,000 iterations, fixed seed 42, replicates resampled within each concentration among the wells retained in the final fit; the outlier screening and point removal are not repeated in each resample) as a complementary check.
* **No silent failures.** Every skipped or rejected block is reported on the console and in a `NotReached_*.txt` note. A drug name that is not in the registry is never fitted with an invented concentration.

## Requirements and installation

```bash
git clone https://github.com/jmgheredia/PyIC50.git
cd PyIC50
pip install numpy pandas matplotlib scipy openpyxl xlrd
```

`xlrd` is used by pandas to read `.xls` files; `openpyxl` reads and writes `.xlsx`.

## Quick start

1. Put `PyIC50.py` in the folder that contains your plate exports (`.xls`) and `Starting_Concentrations.xlsx`.
2. Run:

```bash
python PyIC50.py
```

The script processes every `*.xls` file in its own folder (not recursively) and writes the results there.

## Input conventions

### Plate layout

* A 96-well plate export saved as `.xls`. The 8 x 12 block of readings is located by structure, so extra metadata rows, row labels (A-H) or trailing statistics columns are tolerated.
* Each plate holds **two blocks of four replicate rows**: rows A-D are block 1 and rows E-H are block 2.
* Within a block, **well 1 (column 1) holds the highest concentration**, wells 2-11 are a serial dilution, and **well 12 is the zero-drug control**. The dilution factor comes from the registry (see below); without it, `Config.SERIAL_DILUTION_FACTOR` (default 3, i.e. 1/3) applies. Do not leave well 1 empty or start the series in well 2: all IC50 values would be shifted.
* Detector-saturation markers written as `(+)` in place of a number are accepted.

### File names

```
<cell line(s)>; <drug 1> (<volume>), <drug 2> (<volume>).xls
```

* `;` and `,` are interchangeable separators.
* The first cell line and first drug describe block 1 (rows A-D); the second ones describe block 2 (rows E-H). If the name contains only one cell line (or one drug), both blocks use it.
* `(<volume>)` is the stock volume in microlitres added to well 1. It is used to look up the concentration in the registry. If it is omitted, 2 is assumed.
* Cell-line names are taken literally from the file name, so `RPMI 5000` and `RPMI_5000` are different cell lines.

Examples:

```
RPMI_5000; Docetaxel (2), Gemcitabine (2).xls
Detroit; Erlotinib (2), Erlotinib_2 (2).xls
RPMI_5000_Dil4; Docetaxel (2), Gemcitabine (2).xls
```

Special tokens (case-insensitive):

| Token | Where | Meaning |
|---|---|---|
| `unknown`, `nodrug` | drug position | no compound applied; the block is skipped |
| `Empty` | cell-line position | wells without cells (blank); the block is skipped |

### Starting_Concentrations.xlsx

A spreadsheet in the same folder, named `Starting_Concentrations.xlsx`, with a header row (its text is ignored) followed by columns: drug name, stock volume in microlitres, the concentration in well 1, and, optionally, the serial dilution factor. Results are labelled in uM, so register concentrations in uM.

| Drug | Volume | Concentration | Dilution |
|---|---|---|---|
| Docetaxel | 2 | 20 | 3 |
| Gemcitabine | 2 | 37.99 | 3 |
| Gemcitabine | 20 | 379.94 | 3 |
| CisPt | 8 | 128 | 2 |

* **Dilution** is the factor between consecutive wells: write `3` for 1:3 and `2` for 1:2 (`1:3` written as text is also understood). Type the plain number: Excel turns a typed `1:3` or `1/3` into a time or a date, and the script stops with a message if it finds that. An empty cell, or no fourth column, means `Config.SERIAL_DILUTION_FACTOR` (3). The factor used is written to the `Serial_Dilution_Factor` column of the results.
* The factor belongs to the drug (and volume) row, so two drugs on one plate may have different factors, each applied to its own block.
* Names are matched case-insensitively and the **whole token** must match: `Phenphormin` does not match `CarboPt_phenphormin`. Register a drug combination as its own row.
* One row per volume if the same drug is used at several volumes. If the volume in the file name is not registered, the closest registered volume is used and a warning is printed.
* A drug that matches no row is **not fitted**. It is skipped, listed in `UNREGISTERED_DRUGS_WARNING.txt`, and gets a `NotReached_*.txt` note. Check the console output after each run.

## How each block is analysed

1. The concentration series is built from the registered well-1 concentration and the drug's dilution factor; well 12 is the zero-drug control.
2. Replicate wells are screened for outliers within each concentration column (saturated wells excluded).
3. A first free weighted 4PL fit is made, and the block is checked for a late drop (signal falling only at the highest doses) or, when that fit is poor (R² below 0.90), clearly dead replicates at high dose. If either is found, the lower plateau is anchored from those wells and the fit continues with the gate in step 4 skipped, because the anchor is already per-column evidence of a real response.
4. **Dose-response gate.** Otherwise the median of wells 1-2 is compared with the median of wells 10-11; the high-dose wells must sit at least 15% lower. If that fails, wells 11 and 12 are compared instead (a very potent drug may still be active at the lowest dose): the higher of the two is taken as the true baseline, and if the drop then reaches 15% a fictional top-anchor point is added at that value. Otherwise the block is rejected with `REJECTED_NO_DOSE_RESPONSE`.
5. If the fit is accepted with R² of at least 0.90 it stands as it is. A late-drop curve that is poor and has too few dead wells to anchor from is queued for pass 2, which takes its lower plateau from the median bottom/top ratio of other blocks of the same cell line.
6. The AIC-optimised LOO loop removes anomalous points one at a time, and the reinstatement pass restores any that no longer look anomalous.
7. Uncertainty is estimated parametrically (the interval to report) and by bootstrap (a complementary check).

## Two stocks of the same drug: combined fit

If two stocks of one drug (different concentrations) were run on the same plate, register them as `Drug` and `Drug_2` (or as the same name with two volumes) and put both in one file:

```
Detroit; Erlotinib (2), Erlotinib_2 (2).xls
```

Each block is fitted on its own. If both fits are `OK`, a third fit pools all their wells and is reported as `Erlotinib (Combined)`.

* Use this **only when both stocks are the same compound**. `_2` is just a label for the second stock; it says nothing about which one is more concentrated.
* Only the exact suffix `_2` is recognised. Names such as `Drug_th` or `Drug_batch2` are treated as different drugs and are never pooled, which is what you want when comparing two batches of the same drug.

## Diluted re-reads: saturation correction

If an undiluted plate saturates the detector, a second read of the same plate after dilution can be used. Name the diluted file with `_DilN` after the cell line, keeping everything else identical (same plate layout and same drug part):

```
RPMI_5000; Docetaxel (2), Gemcitabine (2).xls
RPMI_5000_Dil4; Docetaxel (2), Gemcitabine (2).xls
```

* Each file is analysed on its own as usual.
* Saturated wells of the undiluted file that have a non-saturated counterpart at the same well in the diluted file are recovered with an **empirical rescale** (linear fit when at least 4 reference wells exist, median ratio otherwise) fitted on wells where both reads are reliable. The nominal dilution factor is not used, because absorbance is not linear near saturation.
* The rescale is applied only inside the range of the reference wells (**interpolation, never extrapolation**). Other wells keep the imputed ceiling value.
* The result is an extra row `<Drug> (Dil-Corrected)`; the original rows are never replaced. No corrected row is produced when there is nothing to correct or too few reliable wells.

## Outputs

Written to the same folder:

| File | Content |
|---|---|
| `Consolidated_IC50_Results.xlsx` | all fits, one row per fitting step |
| `Plot_<cell line>_<drug>_vol<volume>_<file name>.png` | one plot per fitted curve |
| `NotReached_*.txt` | one per block that produced no fit, with the reason |
| `UNREGISTERED_DRUGS_WARNING.txt` | only if some drug was not in the registry |

### Reading the consolidated table

A block gives one or more rows: `Initial Cleaned Fit`; if its R² is below 0.90, one row for each accepted LOO round (`LOO round #n`); and a `Reinstated` row when points were restored. **The last row of a block is its final model** and the one that carries the bootstrap interval.

| Columns | Meaning |
|---|---|
| `File`, `Cell_Line`, `Drug`, `Stock_Volume_uL`, `Serial_Dilution_Factor`, `Well1_Concentration_uM` | what was fitted |
| `Fit_Status`, `Run`, `Removed_Point`, `N_Removed` | outcome (`OK` or `REJECTED_NO_DOSE_RESPONSE`), the fitting step, the point removed in that step, and total points removed so far |
| `IC50_uM`, `SE_IC50_uM` | IC50 and its standard error |
| `IC50_CI95_lo`, `IC50_CI95_hi` | parametric 95% confidence interval (the interval to report) |
| `IC50_Boot_lo`, `IC50_Boot_hi`, `IC50_Boot_median` | bootstrap 95% interval and median (final row of a block only); indicative, see "Known limitations" |
| `HillSlope`, `Bottom`, `Top` | curve parameters, each with `SE_*` and `*_CI95_lo` / `*_CI95_hi` |
| `R2`, `AIC` | goodness of fit |
| `Fixed_Bottom` | the lower plateau was fixed (anchored) instead of fitted freely |
| `Fallback_Pass2` | fallback flag; see the caveat in "Known limitations" |
| `Top_Anchor_Used` | the dose-response gate fallback added a fictional top-anchor point |
| `Saturation_Corrected`, `N_Wells_Corrected`, `IC50_Delta_vs_Raw_uM` | only on `(Dil-Corrected)` rows: wells recovered and the IC50 shift against the uncorrected fit |

## Configuration

The main parameters are at the top of the `Config` class in `PyIC50.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `SERIAL_DILUTION_FACTOR` | 3.0 | default well N+1 = well N / factor, used when the registry gives no dilution for a drug |
| `MIN_DOSE_RESPONSE_DROP` | 0.15 | minimum drop for the dose-response gate |
| `R2_POOR_FIT` | 0.90 | R² below which a fit is treated as poor |
| `LOO_MAX_REMOVED` | 12 | maximum points removed by the LOO loop |
| `LOO_REINSTATE_SD` | 1.5 | reinstatement tolerance, in SD of the final residuals |
| `N_BOOT`, `BOOT_SEED` | 1000, 42 | bootstrap iterations and seed |
| `SATURATION_TOKENS` | `{"(+)"}` | text that marks a saturated well |

Whether the factor comes from the registry or from `SERIAL_DILUTION_FACTOR`, it rebuilds the concentration series and the fictional anchor points consistently.

## Validation

The fit was compared with the R package `drc` (four-parameter log-logistic model `LL.4`, default unweighted least squares) on simulated plates with a known true IC50: 7 scenarios x 150 datasets, four replicates per dose, eleven 1/3 dilutions from 100 uM plus a zero-drug control. The noise is proportional to the signal plus an additive floor, except in one scenario with constant noise; one scenario adds a gross outlier. "Typical error" is the root-mean-square error of log10(IC50), shown as a percentage.

| Scenario | Typical error, PyIC50 | Typical error, drc | Parametric 95% interval contains the truth, PyIC50 | drc |
|---|---|---|---|---|
| 3% noise | 4.2% | 3.8% | 0.93 | 0.99 |
| 6% noise | 6.2% | 5.7% | 0.95 | 1.00 |
| 10% noise | 10.7% | 10.2% | 0.93 | 0.99 |
| steep curve | 4.2% | 4.2% | 0.96 | 0.97 |
| shallow curve | 16.4% | 20.8% | 0.93 | 0.99 |
| one gross outlier | 6.9% | 14.0% | 0.93 | 0.91 |
| constant noise | 10.9% | 9.6% | 0.95 | 0.97 |

* Accuracy is within 1.3 percentage points of `drc` in five scenarios, and better with a shallow curve and with an outlier, which the automatic screening handles.
* The parametric interval contains the true IC50 in 93-96% of datasets (nominal 95%). The intervals of `drc` cover 91-100% but are up to 3.6 times wider.
* For two plates with the same true IC50, the parametric intervals fail to overlap in 1% of pairs (about 0.6% is expected for a correct 95% interval).
* The bootstrap interval contains the truth in 71-84% of datasets (see "Known limitations").
* The results depend on the noise structure. The weights assume that the variance grows with the signal; the constant-noise scenario shows the behaviour when it does not. The simulation fits single blocks and does not include the rest of the pipeline's decisions on real plates.

The scripts to reproduce it are in `validation/simulation`.

## Known limitations

* Input must be `.xls` (not `.xlsx`), a standard 96-well plate with two 4-row blocks of 12 columns, with the highest concentration in well 1. Both blocks of a plate are assumed to follow the dilution registered for their own drug; the script cannot check it against the plate.
* Metadata comes from file names: a spelling difference creates a separate cell line or an unregistered drug. Check the console output and `UNREGISTERED_DRUGS_WARNING.txt` after each run.
* The `_2` combined fit assumes both stocks are the same compound measured under the same conditions; the script cannot check that.
* Saturation correction recovers only wells inside the range covered by reliable reference wells. On plates where most wells saturate (for example a reader that reports `(+)` over most of the plate), few or none can be recovered.
* The rules that look at fixed well positions (dose-response gate, anchoring, plateau checks) were tuned with 1/3 dilutions. They run with other factors but cover a different concentration range, so inspect the first results when you use a different dilution factor.
* `Fallback_Pass2` is also set on some leave-one-out rows that follow a pre-cleaned initial fit, so do not read it as "pass 2 was used" on its own. Combine it with `Fixed_Bottom`.
* For a drug not registered at 2 microlitres, the console may print `No entry ... vol=2`; the concentration actually used is the one registered for the volume written in the file name.
* The bootstrap interval is optimistic with few replicates: with four replicates per dose it contained the true IC50 in about 79% of simulated datasets (71-84% by scenario). It is also conditional on the quality-control decisions: it resamples the wells retained in the final fit and does not repeat the outlier screening or point removal in each resample. Use the parametric interval (`IC50_CI95_lo`, `IC50_CI95_hi`) as the reference and read the bootstrap interval as a rough check.
* Both intervals describe how precisely one plate defines the curve. They do not include experiment-to-experiment variability (cell passage, seeding density, reagent batches). To compare conditions, use the spread of the IC50 across independent experiments.
* There is no automated test suite in the repository yet.

## Citation

Please cite the archived release: [DOI to be added after the first Zenodo release].

## License

MIT. See [LICENSE](LICENSE).

## Author and contributions

Jose Manuel Garcia Heredia, Universidad de Sevilla.

Bug reports, questions and suggestions are welcome through the [issue tracker](https://github.com/jmgheredia/PyIC50/issues).
