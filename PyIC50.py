import os
import sys
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, nnls
from scipy import stats as scipy_stats
from scipy.stats import spearmanr
import warnings
from scipy.optimize import OptimizeWarning

# On Windows, stdout/stderr default to the system codepage (e.g. cp1252 on
# Spanish/Western-European installs), which cannot encode symbols such as
# "→" or "−" used throughout this script's console output. Forcing UTF-8
# here -- rather than avoiding those symbols -- prevents a crash regardless
# of the user's locale or terminal.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
#  1. GLOBAL CONFIGURATION & METADATA
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    """Centralized, scientifically justified configuration parameters."""
    # Serial dilution of the plate: well N+1 = well N / SERIAL_DILUTION_FACTOR
    # (3.0 = 1/3, 2.0 = 1/2, 4.0 = 1/4 ...). Well 1 comes from
    # Starting_Concentrations; wells 2-11 follow from it; well 12 is the
    # zero-drug control. This is only the DEFAULT: a drug whose row in
    # Starting_Concentrations.xlsx has a 4th column (dilution factor) uses
    # that value instead.
    SERIAL_DILUTION_FACTOR = 3.0

    # Fitting constraints & thresholds
    R2_POOR_FIT = 0.90            # Minimum R² to consider a standard fit acceptable
    AIC_MIN_IMPROVEMENT = -2.0    # AIC must decrease by at least 2.0 to justify point removal
    
    # Biological decision rules
    DROP_THRESHOLD = 0.65         # Well signal must drop below 65% of control to declare a drop
    FLAT_THRESHOLD = 0.80         # Mid-plateau signal must stay above 80% of control to confirm flatness
    DEAD_WELL_THRESHOLD = 0.35    # Well below 35% of control is considered biochemically dead
    ANCHOR_DEAD_FRACTION = 0.5    # Minimum fraction of dead replicates to trigger anchoring
    
    # LOO loop bounds
    LOO_MAX_REMOVED = 12          # Maximum allowable dropped points (25% of a 48-point dataset)
    LOO_SD_THRESHOLD = 0.15       # Replicate SD threshold (15% of control) to prioritize a column
    EXTREME_COLS = {0, 1, 9, 10}  # Columns carrying highest/lowest active doses

    # Low-dose biological plateau filter (single pass, non-recursive; see
    # _run_loo_fits). Reference and check columns are deliberately disjoint
    # so a candidate outlier well cannot influence its own yardstick.
    BIO_PREFILTER_REFERENCE_COLS = {7, 8}  # Used to estimate the living-cell plateau
    BIO_PREFILTER_CHECK_COLS = {9, 10}     # Low-dose columns screened against it
    BIO_PREFILTER_DROP_FRACTION = 0.70     # Flag wells below 70% of the reference plateau
    BIO_PREFILTER_MAX_REF_CV = 0.20        # Abstain if the reference itself has CV > 20%
    BIO_PREFILTER_MAX_REMOVED = 2          # Hard cap on points removed by this filter

    # Post-LOO reinstatement check (see _run_loo_fits)
    LOO_REINSTATE_SD = 1.5        # Restore a dropped point if within this many SD of the final fit

    # Dose-response direction gate (see _run_loo_fits, step 0)
    DOSE_GATE_HIGH_COLS = {0, 1}     # Highest tested concentrations
    DOSE_GATE_LOW_COLS = {9, 10}     # Lowest active / near-baseline concentrations
    MIN_DOSE_RESPONSE_DROP = 0.15    # Highest dose must sit >=15% below the baseline plateau
    # Fallback reference when DOSE_GATE_LOW_COLS still hasn't recovered to a
    # true no-drug baseline (potent drug/sensitive line): last real dilution
    # point vs. the true zero-drug control -- see _run_loo_fits, step 0.
    DOSE_GATE_FALLBACK_REF_COLS = (10, 11)
    
    # Realistic pharmacological parameter bounds
    HILL_SLOPE_MIN = -5.0         # Lower Hill bound (standard toxicological limit)
    HILL_SLOPE_MAX = 5.0          # Upper Hill bound
    
    # Robustness & uncertainty estimation
    N_BOOT = 1000                 # Peer-reviewed publishing standard for bootstrap iterations
    BOOT_SEED = 42                # Global seed for pseudo-random number generator
    
    # QC Pipetting thresholds
    CONTROL_PLATEAU_THRESHOLD = 0.60  # Control col must be > 60% of intermediate plateau
    PLATEAU_COLS = [7, 8, 9]          # Columns used to estimate high-signal plateau
    
    # Plotting specifications
    X_LIM_MIN = 1e-5
    X_LIM_MAX = 1500

    # Detector-saturation handling. Some plate readers flag a well that
    # exceeded the instrument's linear detection range with a censoring
    # marker instead of a number (e.g. "( + )"). These are real, biologically
    # meaningful readings (very likely near/at the living-cell plateau) --
    # not missing data -- so they are imputed rather than discarded. Tokens
    # are matched after stripping whitespace and lowercasing, so "( + )",
    # "(+)" and "( +)" all match.
    SATURATION_TOKENS = {"(+)"}

    # Tokens used in the drug segment of a filename to mark leftover wells
    # that never received any compound (e.g. 4 spare wells on a plate laid
    # out for a different drug count). Matched case-insensitively after
    # stripping a volume-in-parens suffix. A block whose drug token matches
    # one of these has no dose-response to fit and is skipped entirely,
    # rather than being assigned a fabricated placeholder concentration.
    NO_DRUG_TOKENS = {"unknown", "nodrug"}

    # Tokens used in the cell-line segment of a filename to mark wells that
    # never received any cells (a blank/background control block). Matched
    # case-insensitively against the whole cell-line token. A block whose
    # cell line matches one of these has no dose-response to fit -- it is
    # background/instrument noise, not biology -- and is skipped entirely
    # rather than treated as a real second cell line.
    NO_CELL_LINE_TOKENS = {"empty"}

# Set deterministic random seed
np.random.seed(Config.BOOT_SEED)


# ─────────────────────────────────────────────────────────────────────────────
#  2. MATHEMATICAL MODELS & EXPERIMENTAL OUTLIER DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def log_4pl(x, bottom, top, log_ic50, hill_slope):
    """Standard 4-Parameter Logistic (4PL) Model."""
    return bottom + (top - bottom) / (1 + 10**((log_ic50 - x) * hill_slope))

def compute_aic(ss_res, n, k):
    """
    Computes the Akaike Information Criterion (AIC) for non-linear least squares.
    AIC = n * ln(RSS / n) + 2 * k
    """
    if n <= k + 1:
        return np.inf
    return n * np.log(ss_res / n) + 2 * k

def detect_outliers_mad(y_vals, threshold=2.5):
    """Outlier detection utilizing Median Absolute Deviation (MAD)."""
    y_vals = np.array(y_vals)
    median = np.median(y_vals)
    diff = np.abs(y_vals - median)
    mad = np.median(diff)
    if mad == 0:
        std = np.std(y_vals)
        if std == 0:
            return np.zeros(len(y_vals), dtype=bool)
        z_scores = 0.6745 * (y_vals - median) / (1.4826 * std)
    else:
        z_scores = 0.6745 * (y_vals - median) / (1.4826 * mad)
    return np.abs(z_scores) > threshold

def detect_experimental_outliers(X, Y, popt, pcov, confidence=0.95):
    """
    Detects individual experimental outliers by projecting a vertical prediction 
    band at each concentration. This catches bimodal replicate drops (e.g., 50/50 
    pipetting splits) where MAD fails due to lack of a single median-based outlier.
    """
    n = len(X)
    k = len(popt)
    if n <= k:
        return np.zeros(n, dtype=bool)

    y_pred = log_4pl(np.log10(X), *popt)
    residuals = Y - y_pred
    ss_res = np.sum(residuals**2)
    rse = np.sqrt(ss_res / (n - k)) if (n - k) > 0 else 0.0

    t_crit = scipy_stats.t.ppf((1 + confidence) / 2.0, max(n - k, 1))

    # Numerical Jacobian calculation: dy/dp for each parameter p at each point x
    eps = 1e-5
    jacobian = np.zeros((n, k))
    for i in range(k):
        p_perturbed = np.copy(popt)
        p_perturbed[i] += eps
        y_perturbed = log_4pl(np.log10(X), *p_perturbed)
        jacobian[:, i] = (y_perturbed - y_pred) / eps

    outliers = np.zeros(n, dtype=bool)
    for idx in range(n):
        j_x = jacobian[idx, :]
        param_var = j_x.dot(pcov).dot(j_x.T)
        pred_var = param_var + (rse ** 2)
        pred_se = np.sqrt(max(pred_var, 1e-10))

        distance = np.abs(Y[idx] - y_pred[idx])
        if distance > t_crit * pred_se:
            outliers[idx] = True

    return outliers


# ─────────────────────────────────────────────────────────────────────────────
#  3. FILE PARSING & DATA MANIPULATION
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_token(val):
    """Whitespace/case-insensitive normalization for matching marker cells."""
    if not isinstance(val, str):
        return None
    return re.sub(r"\s+", "", val).lower()


def extract_plate_grid(file_path, n_wells_per_row=12, min_rows=8):
    """
    Loads a raw plate export and extracts the well grid, tolerating
    metadata that real spectrophotometer software often adds around the
    96 raw values -- e.g. a leading row-label column (A-H), a metadata
    row (experiment name, date, plate ID) above the data, or detector-
    saturation markers such as "( + )" in place of an out-of-range value.

    Rather than trusting fixed row/column positions (which silently
    shift every well if anything extra is present), this locates the
    grid structurally: a "data row" is any row with at least
    n_wells_per_row cells that are either numeric or a recognized
    saturation marker, and a "data column" is any column meeting that
    same test across every candidate data row. Non-numeric label columns
    fail it and are dropped automatically. A leftover row that is just
    the literal index sequence (1..N) -- e.g. a "1,2,...,12" column-
    header row that survived because it's numeric -- is recognized as a
    positional label, not a reading, and dropped too.

    Saturated cells are real, biologically meaningful readings (the true
    value is at or above the instrument's ceiling) rather than missing
    data, so they are imputed with the highest reliably-read value found
    elsewhere in the same file -- a conservative, file-specific estimate
    of that ceiling -- instead of being discarded. The returned mask lets
    callers track which points were imputed (e.g. to exclude them from
    outlier detection or mark them distinctly on plots).

    Returns (df_block, sat_mask, ceiling_value, n_saturated).
    """
    df_raw = pd.read_excel(file_path, header=None)
    df_num = df_raw.apply(pd.to_numeric, errors="coerce")
    sat_mask_raw = df_raw.map(lambda v: _normalize_token(v) in Config.SATURATION_TOKENS)

    # A cell "counts" toward the data grid if it's numeric OR a
    # recognized saturation marker -- both are real well readings, just
    # one is off-scale. A label/metadata cell is neither.
    present = df_num.notna() | sat_mask_raw

    row_counts = present.sum(axis=1)
    candidate_rows = df_num.index[row_counts >= n_wells_per_row]
    if len(candidate_rows) == 0:
        raise ValueError(f"no row with >= {n_wells_per_row} numeric/saturated values found")

    col_is_present = present.loc[candidate_rows].all(axis=0)
    data_cols = df_num.columns[col_is_present]
    if len(data_cols) < n_wells_per_row:
        raise ValueError(
            f"only {len(data_cols)} fully-populated columns found "
            f"(need >= {n_wells_per_row} for a standard plate row)"
        )
    # Well data is always contiguous starting at Well 1; if extra numeric
    # columns survive (e.g. a trailing per-row average column), keep the
    # leftmost n_wells_per_row.
    data_cols = data_cols[:n_wells_per_row]

    df_block = df_num.loc[candidate_rows, data_cols]
    sat_block = sat_mask_raw.loc[candidate_rows, data_cols]
    df_block.columns = range(n_wells_per_row)
    sat_block.columns = range(n_wells_per_row)
    df_block = df_block.reset_index(drop=True)
    sat_block = sat_block.reset_index(drop=True)

    index_seq = np.arange(1, n_wells_per_row + 1)
    is_index_row = df_block.eq(index_seq).all(axis=1)
    df_block = df_block[~is_index_row].reset_index(drop=True)
    sat_block = sat_block[~is_index_row].reset_index(drop=True)

    if len(df_block) < min_rows:
        raise ValueError(
            f"only {len(df_block)} usable data row(s) found after filtering "
            f"non-numeric/label content (need >= {min_rows})"
        )

    n_saturated = int(sat_block.values.sum())
    ceiling_value = np.nan
    if n_saturated > 0:
        real_vals = df_block.values[~sat_block.values]
        if real_vals.size == 0:
            raise ValueError(
                "every well is flagged as saturated -- no real reading in "
                "the file to estimate a detector ceiling from"
            )
        ceiling_value = float(np.nanmax(real_vals))
        df_block = df_block.mask(sat_block, ceiling_value)

    return df_block, sat_block, ceiling_value, n_saturated


_DIL_SUFFIX_RE = re.compile(r'_dil(\d+)', re.IGNORECASE)


def _extract_dilution_factor(file_name):
    """
    Reads the nominal dilution factor (e.g. 4 from "..._Dil4;...") off a
    plate filename's cell-line segment -- this lab's convention marks a
    diluted re-read of the same physical plate there rather than on the
    drug segment (see analyze_drug_dilution_corrected), because matching
    against Starting_Concentrations.xlsx worked worse the other way.
    Returns None for a filename with no such marker -- the concentrated/
    undiluted read.
    """
    m = _DIL_SUFFIX_RE.search(file_name)
    return int(m.group(1)) if m else None


def _dilution_pair_key(file_name):
    """
    Filename with any '_DilN' marker removed, so a concentrated file and
    its diluted re-read of the identical plate collapse onto the same
    grouping key for pairing (see analyze_drug_dilution_corrected).
    """
    return _DIL_SUFFIX_RE.sub('', file_name)


_RATIO_RE = re.compile(r"^\s*1\s*[:/]\s*(\d+(?:[.,]\d+)?)\s*$")


def _parse_dilution_factor(raw, drug_label):
    """
    Reads the optional 4th column of Starting_Concentrations.xlsx: the
    serial dilution factor of that drug's plate block (3 = 1/3, 2 = 1/2 ...).

    Accepts a number >= 1.01 (3, 2, 4, 2.5) or a text ratio "1:3" / "1/3".
    Returns None for an empty cell (the caller then falls back to
    Config.SERIAL_DILUTION_FACTOR). Anything else raises ValueError and
    names the row, because a guessed factor would silently shift every
    concentration of that drug.

    Typing 1:3 or 1/3 straight into an Excel cell is turned into a time or a
    date by Excel itself; that case is detected and reported rather than
    misread. Writing the plain number (3) avoids the problem.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    if isinstance(raw, str):
        if not raw.strip():
            return None
        m = _RATIO_RE.match(raw)
        if m:
            raw = float(m.group(1).replace(",", "."))
        else:
            try:
                raw = float(raw.strip().replace(",", "."))
            except ValueError:
                raise ValueError(
                    f"Starting_Concentrations.xlsx, '{drug_label}': dilution factor "
                    f"'{raw}' not understood. Write 3 for a 1:3 dilution, 2 for 1:2.")
    elif not isinstance(raw, (int, float, np.integer, np.floating)) or isinstance(raw, bool):
        raise ValueError(
            f"Starting_Concentrations.xlsx, '{drug_label}': the dilution-factor cell was read "
            f"as {type(raw).__name__} ('{raw}'). Excel turns a typed 1:3 or 1/3 into a time or "
            f"a date. Write just the number (3 for 1:3, 2 for 1:2).")
    factor = float(raw)
    if not np.isfinite(factor) or factor <= 1.0:
        raise ValueError(
            f"Starting_Concentrations.xlsx, '{drug_label}': dilution factor {raw} must be "
            f"greater than 1 (3 for a 1:3 dilution, 2 for 1:2).")
    return factor


def parse_filename(filename, conc_dict, dilution_dict=None):
    """
    Parses plate design configuration from the filename.

    The cell-line segment is whatever precedes the first token that
    carries a volume-in-parens suffix (e.g. "CarboPt (2)") -- this lab's
    convention never puts that suffix on a cell-line name, so it marks
    the start of the drug segment structurally, regardless of whether
    that token's drug name happens to be registered yet.

    An earlier version anchored on the first *registered* drug name
    instead. That broke as soon as an *earlier* drug token in the same
    filename wasn't registered: e.g. "Line1; Gemcitabina (2), CarboPt
    (2)" with only "Gemcitabine" (not "Gemcitabina") registered would
    anchor on "CarboPt" and silently swallow "Gemcitabina (2)" into
    `cell_lines` instead of flagging it as an unregistered drug.

    Labs are not consistent about whether they separate multiple cell
    lines -- or the cell-line segment from the drug segment -- with a
    comma or a semicolon (e.g. "LineA; LineB; Drug" vs "LineA,LineB;
    Drug"), so both are accepted as equivalent separators.
    """
    base_name = os.path.basename(filename)
    name_no_ext = os.path.splitext(base_name)[0]

    region_offset = None
    pos = 0
    for token in re.split(r"([;,])", name_no_ext):
        if token in (";", ","):
            pos += len(token)
            continue
        # A volume-in-parens suffix anchors the drugs region, same as
        # before. A bare NO_DRUG_TOKENS match (e.g. "nodrug") anchors it
        # too, even without a volume suffix -- otherwise a no-drug
        # condition token with nothing else before it in the drugs
        # segment gets swallowed into the cell-line segment instead,
        # silently turning a control block into a fabricated second
        # cell line (see build_assignments' drug2 fallback).
        if re.search(r"\(\d+\)", token) or token.strip().lower() in Config.NO_DRUG_TOKENS:
            region_offset = pos
            break
        pos += len(token)

    if region_offset is not None:
        lines_part = name_no_ext[:region_offset]
        drugs_region = name_no_ext[region_offset:]
    elif ";" in name_no_ext:
        # No token anywhere carries a volume suffix -- there is no
        # structural anchor to locate the drug segment by, so fall back
        # to the older convention: cell line(s) before the first ';',
        # drug(s) after it. (Splitting the *whole* filename into
        # cell-line tokens here, as if no drug segment existed at all,
        # was tried and is wrong: it swallows the drug segment into
        # `cell_lines` and drops it entirely.)
        lines_part, drugs_region = name_no_ext.split(";", 1)
        region_offset = len(lines_part) + 1
    else:
        return None, None

    # Strip an optional '_DilN' marker (e.g. "RPMI_sg8-18-12_Dil4") from
    # each cell-line token so the reported cell line is identical whether
    # this file is the concentrated read or its diluted re-read of the
    # same plate -- required for _dilution_pair_key's grouping to find
    # them as the same block (see analyze_drug_dilution_corrected).
    cell_lines = [
        re.sub(r'_dil\d+$', '', l.strip(), flags=re.IGNORECASE)
        for l in re.split(r"[;,]", lines_part) if l.strip()
    ]
    while len(cell_lines) < 1:
        cell_lines.append("CellLine_1")
    cell_lines = cell_lines[:2]

    drugs = (_parse_drug_tokens(drugs_region, region_offset, conc_dict, dilution_dict)
             if drugs_region else [])

    while len(drugs) < 1:
        drugs.append({
            'name': 'UnknownDrug_1',
            'volume': 2,
            'base_conc': 100.0,
            'exact_conc': False,
            'search_index': 999
        })

    return cell_lines, drugs[:2]

def _parse_drug_tokens(drugs_region, region_offset, conc_dict, dilution_dict=None):
    """
    Splits the drug segment of a filename into one condition per
    comma/semicolon-separated token and resolves each token to exactly
    one drug entry independently, rather than scanning the whole region
    for substring matches in one pass.

    Scanning the whole region at once lets a short registered key match a
    token it doesn't fully belong to. With only "carbopt" and
    "phenphormin" registered (no combined "carbopt_phenphormin" entry),
    the token "carbopt_phenphormin" would match "carbopt" as one hit AND
    "phenphormin" as a second, separate hit -- alongside the genuine
    standalone "carbopt" earlier in the filename -- producing 3 raw drug
    candidates for what are only 2 plate blocks. Truncating to 2 then
    silently kept two duplicate "Carbopt" entries and dropped
    "Phenphormin" entirely: an unregistered drug *combination* was
    mislabeled as a duplicate of a single drug, and one plate block's
    fit overwrote the other's on disk.

    Each token is resolved on its own instead: a registered key only
    counts as *the* drug for a token if it matches the token's full text
    (ignoring a volume-in-parens suffix) -- not just a fragment of it.
    Otherwise the token -- no match, or a match covering only part of it,
    e.g. just "carbopt" out of "carbopt_phenphormin" -- is treated as its
    own unregistered condition (or the "no compound applied" case, see
    Config.NO_DRUG_TOKENS), named after its own literal text, rather than
    silently collapsed onto whichever shorter key happened to match a
    fragment of it.
    """
    # A '_' right after a volume-in-parens suffix (e.g. "...(2)_ next...")
    # separates two conditions, same as ',' or ';' -- as opposed to a bare
    # '_' elsewhere, which is how a combination names its own components
    # (e.g. "docetaxel_phloridzin"). Without this, a filename joining two
    # conditions with '_' instead of ','/';' collapses them into a single
    # token: the volume regex below truncates at the *first* "(N)" it
    # finds, silently discarding everything after it -- no placeholder,
    # no NotReached note, the second condition simply never existed.
    drugs_region = re.sub(r"(\(\d+\))_", r"\1;", drugs_region)

    result = []
    pos = 0
    for token in re.split(r"([;,])", drugs_region):
        if token in (";", ","):
            pos += len(token)
            continue
        token_start = pos
        pos += len(token)
        stripped = token.strip()
        if not stripped:
            continue

        # Only a *parenthesized* number is a volume suffix -- a bare digit
        # match here (e.g. the "11" in "erlotinib_11") could just as easily
        # be part of the compound's own name, so name and volume must be
        # split at the parens, not by stripping every digit in the token.
        vol_match = re.search(r"\((\d+)\)", stripped)
        if vol_match:
            volume = int(vol_match.group(1))
            core_text = stripped[:vol_match.start()].strip()
        else:
            volume = 2
            core_text = stripped

        matches = _extract_drugs_from_string(core_text, conc_dict) if core_text else []
        full_match = next(
            (m for m in matches
             if m["search_index"] == 0 and len(m["name"]) == len(core_text)),
            None
        )

        if full_match is not None:
            entry = conc_dict[core_text.lower()]
            if isinstance(entry, dict):
                if volume in entry:
                    used_vol = volume
                    full_match["base_conc"] = entry[volume]
                    full_match["exact_conc"] = True
                else:
                    used_vol = min(entry.keys(), key=lambda v: abs(v - volume))
                    full_match["base_conc"] = entry[used_vol]
                    full_match["exact_conc"] = True
                    print(f"    [Warning] No entry for '{core_text}' vol={volume} uL "
                          f"-> using vol={used_vol} uL ({full_match['base_conc']} uM) instead.")
                # None when the registry row has no dilution factor: the
                # caller then applies Config.SERIAL_DILUTION_FACTOR.
                full_match["dilution_factor"] = (dilution_dict or {}).get(core_text.lower(), {}).get(used_vol)
            full_match["volume"] = volume
            full_match["search_index"] = region_offset + token_start
            result.append(full_match)
            continue

        name = core_text or "UnknownDrug"

        if name.lower() in Config.NO_DRUG_TOKENS:
            result.append({
                'name': name.capitalize(),
                'volume': volume,
                'base_conc': None,
                'exact_conc': False,
                'search_index': region_offset + token_start,
                'is_no_drug': True,
            })
            continue

        # base_conc=100.0 here is a marker value only -- the caller checks
        # is_unregistered_placeholder and skips fitting entirely rather than
        # ever using it, so a fabricated concentration never reaches a plot
        # or a result row (see the main pipeline loop).
        result.append({
            'name': name.capitalize(),
            'volume': volume,
            'base_conc': 100.0,
            'exact_conc': False,
            'is_unregistered_placeholder': True,
            'search_index': region_offset + token_start,
        })
    return sorted(result, key=lambda d: d["search_index"])


def _extract_drugs_from_string(text, conc_dict):
    """
    Extracts known drugs and maps concentration reference points.

    Keys are matched longest-first so that an explicit combination entry
    registered in the concentration file (e.g. "cisPt+lenumlostat" as its
    own row, with its own Well-1 concentration) takes priority over its
    individual components.

    Once a key claims a character span within the filename, any shorter
    key whose match falls *inside* that already-claimed span is skipped
    entirely rather than added as a separate drug. Without this check,
    a registered combination such as "cisPt+lenumlostat" would be
    silently split into two independent entries — the full combination
    AND a duplicate "Lenumlostat" match found as a substring inside it —
    corrupting the well-block assignment for that plate.
    """
    text_lower = text.lower()
    found_drugs = []
    claimed_spans = []  # (start, end) ranges already claimed by a longer/earlier key
    sorted_keys = sorted(conc_dict.keys(), key=len, reverse=True)

    for official_key in sorted_keys:
        key_len = len(official_key)
        for m in re.finditer(re.escape(official_key), text_lower):
            start_idx = m.start()
            end_idx = start_idx + key_len

            # Skip matches that fall inside a span already claimed by a
            # longer (earlier-processed) key — e.g. "lenumlostat" found
            # inside an already-matched "cispt+lenumlostat".
            if any(start_idx < c_end and end_idx > c_start
                   for c_start, c_end in claimed_spans):
                continue

            sub_str = text_lower[end_idx:]
            match_vol = re.search(r"\(?\s*(\d+)\s*\)?", sub_str)
            vol = 2
            if match_vol and match_vol.start() < 15:
                vol = int(match_vol.group(1))

            entry = conc_dict[official_key]
            if isinstance(entry, dict):
                if vol in entry:
                    base_conc_val = entry[vol]
                    exact_conc = True
                else:
                    closest_vol = min(entry.keys(), key=lambda v: abs(v - vol))
                    base_conc_val = entry[closest_vol]
                    exact_conc = True
                    print(f"    [Warning] No entry for '{official_key}' vol={vol} µL "
                          f"-> using vol={closest_vol} µL ({base_conc_val} µM) instead.")
            else:
                base_conc_val = entry
                exact_conc = False

            found_drugs.append({
                'name': official_key.capitalize(),
                'volume': vol,
                'base_conc': base_conc_val,
                'exact_conc': exact_conc,
                'search_index': start_idx
            })
            claimed_spans.append((start_idx, end_idx))

    found_drugs = sorted(found_drugs, key=lambda x: x['search_index'])
    filtered = []
    for d in found_drugs:
        overlap = False
        for fd in filtered:
            if (abs(fd['search_index'] - d['search_index']) < 5
                    and fd['name'] == d['name']
                    and fd['volume'] == d['volume']):
                overlap = True
                break
        if not overlap:
            filtered.append(d)

    return sorted(filtered, key=lambda x: x['search_index'])

def build_assignments(cell_lines, drugs, df_clean, sat_clean):
    """Splits raw plate layout into comparative experimental blocks."""
    n_lines = len(cell_lines)
    n_drugs = len(drugs)

    line1 = cell_lines[0]
    line2 = cell_lines[1] if n_lines >= 2 else cell_lines[0]
    drug1 = drugs[0]
    drug2 = drugs[1] if n_drugs >= 2 else drugs[0]

    return [
        (line1, drug1, df_clean.iloc[0:4], sat_clean.iloc[0:4]),
        (line2, drug2, df_clean.iloc[4:8], sat_clean.iloc[4:8]),
    ]

def create_anchor_points(df_valid, fict_conc, anchor_value):
    """
    Generates a pseudo 'Well -1' per row at a caller-chosen fictional
    concentration, to anchor a plateau the real data doesn't pin down on
    its own. Placing fict_conc above every tested dose anchors Bottom
    (the original, still-default use here); placing it below every tested
    dose -- lower than pseudo_zero_conc -- anchors Top instead (see the
    Dose-Response Gate fallback in _run_loo_fits), for a drug potent
    enough that even the lowest tested dose hasn't recovered to a true
    no-drug baseline.
    """
    fict_rows = []
    unique_rows = sorted(df_valid["row_idx"].unique())
    for r_idx in unique_rows:
        fict_rows.append({
            "orig_idx": -1,
            "row_idx": r_idx,
            "col_idx": -1,
            "concentration": fict_conc,
            "value": anchor_value,
            "type": "fictional",
            "is_mad_outlier": False,
            "is_fictional": True,
            "is_saturated": False,
        })
    return pd.DataFrame(fict_rows)


# ─────────────────────────────────────────────────────────────────────────────
#  4. WEIGHTED MODEL FITTING & INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def run_single_fit(X, Y, init_conc, control_mean, pseudo_zero_conc, fixed_bottom=None,
                   _first_stage=False):
    """
    Fits the 4PL curve using Weighted Least Squares (WLS).

    Weights come from a variance model, var = a + b * fitted_value^2, estimated from
    the residuals of a first unweighted fit, and the covariance is rescaled by the
    residual variance (absolute_sigma=False). Taking the SD of only four replicates
    per dose as exact (the earlier scheme) made the weights noisy and the confidence
    intervals too narrow; see validation/simulation.
    """
    if _first_stage:
        sigma = np.ones_like(Y, dtype=float)
    else:
        first_popt = run_single_fit(X, Y, init_conc, control_mean, pseudo_zero_conc,
                                    fixed_bottom, _first_stage=True)[0]
        fitted = log_4pl(np.log10(X), *first_popt)
        design = np.column_stack([np.ones_like(fitted), fitted ** 2])
        coef, _ = nnls(design, (Y - fitted) ** 2)
        sigma = np.sqrt(np.maximum(design @ coef, 1e-8))

    try:
        with np.errstate(over='ignore', invalid='ignore'):
            if fixed_bottom is not None:
                def log_4pl_fixed(x, top, log_ic50, hill_slope):
                    return log_4pl(x, fixed_bottom, top, log_ic50, hill_slope)
                p0 = [control_mean, np.log10(init_conc / 10.0), -1.0]
                bounds = (
                    [control_mean * 0.7, np.log10(pseudo_zero_conc), Config.HILL_SLOPE_MIN],
                    [control_mean * 1.3, np.log10(init_conc * 20.0),  Config.HILL_SLOPE_MAX]
                )
                popt_free, pcov_free = curve_fit(log_4pl_fixed, np.log10(X), Y,
                                                 p0=p0, bounds=bounds, sigma=sigma,
                                                 absolute_sigma=False, maxfev=15000)
                popt = np.array([fixed_bottom, popt_free[0], popt_free[1], popt_free[2]])
                pcov = np.zeros((4, 4))
                pcov[1:, 1:] = pcov_free
            else:
                bottom_guess = np.clip(np.percentile(Y, 5), 0, control_mean * 0.5)
                p0 = [bottom_guess, control_mean, np.log10(init_conc / 10.0), -1.0]
                bounds = (
                    [0.0, control_mean * 0.5, np.log10(pseudo_zero_conc), Config.HILL_SLOPE_MIN],
                    [control_mean * 1.1, control_mean * 1.5, np.log10(init_conc * 20.0), Config.HILL_SLOPE_MAX]
                )
                popt, pcov = curve_fit(log_4pl, np.log10(X), Y,
                                       p0=p0, bounds=bounds, sigma=sigma,
                                       absolute_sigma=False, maxfev=15000)
    except (RuntimeError, OptimizeWarning):
        # Fallback to unweighted Ordinary Least Squares (OLS)
        if fixed_bottom is not None:
            def log_4pl_fixed(x, top, log_ic50, hill_slope):
                return log_4pl(x, fixed_bottom, top, log_ic50, hill_slope)
            p0 = [control_mean, np.log10(init_conc / 10.0), -1.0]
            bounds = (
                [control_mean * 0.7, np.log10(pseudo_zero_conc), Config.HILL_SLOPE_MIN],
                [control_mean * 1.3, np.log10(init_conc * 20.0),  Config.HILL_SLOPE_MAX]
            )
            popt_free, pcov_free = curve_fit(log_4pl_fixed, np.log10(X), Y,
                                             p0=p0, bounds=bounds, maxfev=10000)
            popt = np.array([fixed_bottom, popt_free[0], popt_free[1], popt_free[2]])
            pcov = np.zeros((4, 4))
            pcov[1:, 1:] = pcov_free
        else:
            bottom_guess = np.clip(np.percentile(Y, 5), 0, control_mean * 0.5)
            p0 = [bottom_guess, control_mean, np.log10(init_conc / 10.0), -1.0]
            bounds = (
                [0.0, control_mean * 0.5, np.log10(pseudo_zero_conc), Config.HILL_SLOPE_MIN],
                [control_mean * 1.1, control_mean * 1.5, np.log10(init_conc * 20.0), Config.HILL_SLOPE_MAX]
            )
            popt, pcov = curve_fit(log_4pl, np.log10(X), Y,
                                   p0=p0, bounds=bounds, maxfev=10000)

    residuals = Y - log_4pl(np.log10(X), *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Y - np.mean(Y))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return popt, r2, residuals, pcov


def _compute_se_and_ci(popt, pcov, n_points, alpha=0.05):
    """Calculates Parametric SE and 95% Confidence Intervals."""
    def safe_pow10(exp):
        try:
            if np.isnan(exp) or np.isinf(exp) or abs(exp) > 300:
                return np.nan
            return float(10**exp)
        except Exception:
            return np.nan

    n_free = 4 - (1 if pcov[0, 0] == 0 else 0)
    df = max(n_points - n_free, 1)
    t_crit = scipy_stats.t.ppf(1 - alpha / 2, df)

    diag = np.clip(np.diag(pcov), 0, None)
    se = np.sqrt(diag)

    ic50 = safe_pow10(popt[2])
    se_ic50 = (np.log(10) * ic50 * se[2]) if ic50 is not np.nan else np.nan

    return {
        "SE_Bottom":    float(se[0]),
        "SE_Top":       float(se[1]),
        "SE_log_IC50":  float(se[2]),
        "SE_HillSlope": float(se[3]),
        "SE_IC50_uM":   se_ic50 if not np.isnan(se_ic50) else np.nan,
        "IC50_CI95_lo": safe_pow10(popt[2] - t_crit * se[2]),
        "IC50_CI95_hi": safe_pow10(popt[2] + t_crit * se[2]),
        "Bottom_CI95_lo": float(popt[0] - t_crit * se[0]),
        "Bottom_CI95_hi": float(popt[0] + t_crit * se[0]),
        "Top_CI95_lo":    float(popt[1] - t_crit * se[1]),
        "Top_CI95_hi":    float(popt[1] + t_crit * se[1]),
        "Hill_CI95_lo":   float(popt[3] - t_crit * se[3]),
        "Hill_CI95_hi":   float(popt[3] + t_crit * se[3]),
    }


def _bootstrap_ic50(df_valid, init_conc, control_mean, pseudo_zero_conc,
                    fixed_bottom=None, n_boot=Config.N_BOOT, seed=Config.BOOT_SEED):
    """Robust Bootstrap pipeline to correctly estimate parameter distribution."""
    rng = np.random.default_rng(seed)
    ic50_boots = []
    col_indices = sorted(df_valid["col_idx"].unique())

    for _ in range(n_boot):
        boot_rows = []
        for col in col_indices:
            col_data = df_valid[df_valid["col_idx"] == col]
            if len(col_data) == 0:
                continue
            sample = col_data.sample(n=len(col_data), replace=True,
                                     random_state=int(rng.integers(1e6)))
            boot_rows.append(sample)

        df_boot = pd.concat(boot_rows)
        X_b = df_boot["concentration"].values
        Y_b = df_boot["value"].values

        try:
            popt_b, _, _, _ = run_single_fit(X_b, Y_b, init_conc, control_mean,
                                             pseudo_zero_conc, fixed_bottom=fixed_bottom)
            ic50_boots.append(10**popt_b[2])
        except (RuntimeError, OptimizeWarning):
            continue

    if len(ic50_boots) < 10:
        return np.nan, np.nan, np.nan

    boots = np.array(ic50_boots)
    return (float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)),
            float(np.median(boots)))


# ─────────────────────────────────────────────────────────────────────────────
#  5. LOGICAL DECISION PIPELINE & QC
# ─────────────────────────────────────────────────────────────────────────────
def _prepare_plate_data(df_drug, base_conc, volume, exact_conc=False, sat_drug=None,
                        dilution_factor=None):
    """Initial QC, MAD outlier detection, and pipetting validation."""
    if exact_conc:
        init_conc = base_conc
    else:
        init_conc = base_conc * (volume / 2.0)
    dilution_factor = dilution_factor or Config.SERIAL_DILUTION_FACTOR

    concs_raw = []
    c = init_conc
    for i in range(11):
        concs_raw.append(c)
        c /= dilution_factor
    pseudo_zero_conc = concs_raw[-1] / 100.0

    pts = []
    for r_idx, row in df_drug.iterrows():
        sat_row = sat_drug.loc[r_idx] if sat_drug is not None else None
        for col_idx in range(12):
            conc = concs_raw[col_idx] if col_idx < 11 else pseudo_zero_conc
            pt_type = "dilution" if col_idx < 11 else "control"
            is_saturated = bool(sat_row.iloc[col_idx]) if sat_row is not None else False
            pts.append({
                "orig_idx": len(pts),
                "row_idx": r_idx,
                "col_idx": col_idx,
                "concentration": conc,
                "value": row.iloc[col_idx],
                "type": pt_type,
                "is_mad_outlier": False,
                "is_saturated": is_saturated,
            })
    df_pts = pd.DataFrame(pts)

    # Saturated wells are imputed at a fixed ceiling value, so several of
    # them can legitimately share the same reading -- that is expected
    # censoring, not a pipetting anomaly, and must not be flagged (or
    # allowed to skew the median/MAD) by the outlier test below.
    for col_idx in range(12):
        col_mask = (df_pts["col_idx"] == col_idx) & (~df_pts["is_saturated"])
        y_vals = df_pts.loc[col_mask, "value"].values
        outliers = detect_outliers_mad(y_vals, threshold=2.5)
        row_indices = df_pts.loc[col_mask].index
        for i, is_out in enumerate(outliers):
            if is_out:
                df_pts.at[row_indices[i], "is_mad_outlier"] = True

    plateau_vals = df_pts[df_pts["col_idx"].isin(Config.PLATEAU_COLS)]["value"].values
    control_vals = df_pts[df_pts["col_idx"] == 11]["value"].values

    control_mean_raw = np.mean(control_vals) if len(control_vals) > 0 else np.nan
    plateau_mean     = np.mean(plateau_vals)  if len(plateau_vals) > 0 else np.nan

    systematic_control_outlier = False
    if (not np.isnan(control_mean_raw) and not np.isnan(plateau_mean)
            and plateau_mean > 0
            and control_mean_raw < Config.CONTROL_PLATEAU_THRESHOLD * plateau_mean):
        systematic_control_outlier = True
        ctrl_indices = df_pts[df_pts["col_idx"] == 11].index
        for idx in ctrl_indices:
            df_pts.at[idx, "is_mad_outlier"] = True
        print(f"    [Warning] Col 11 (control) marked as systematic outlier: "
              f"mean={control_mean_raw:.4f} < "
              f"{Config.CONTROL_PLATEAU_THRESHOLD:.0%} of plateau ({plateau_mean:.4f}). "
              f"Control_mean recalculated from cols 7-9.")

    df_valid = df_pts[~df_pts["is_mad_outlier"]].copy()

    if systematic_control_outlier:
        control_mean = plateau_mean
    else:
        control_mean = df_pts[df_pts["type"] == "control"]["value"].mean()

    return df_pts, df_valid, control_mean, init_conc, pseudo_zero_conc


def _is_late_abrupt_drop(df_valid, control_mean):
    """Detects biological drop restricted only to extreme columns (no plateau)."""
    def col_mean(col_indices):
        vals = df_valid.loc[df_valid["col_idx"].isin(col_indices), "value"].values
        return np.mean(vals) if len(vals) > 0 else control_mean

    drop_end  = col_mean([0, 1]) < Config.DROP_THRESHOLD * control_mean
    flat_mid  = col_mean([2, 3]) > Config.FLAT_THRESHOLD * control_mean
    return drop_end and flat_mid


def _should_add_anchor_point(df_valid, control_mean):
    """Evaluates individual replicates at high dose to trigger anchoring."""
    dead_threshold = Config.DEAD_WELL_THRESHOLD * control_mean
    all_dead_vals = []

    for col in [0, 1]:
        col_data = df_valid[df_valid["col_idx"] == col]["value"].values
        if len(col_data) == 0:
            continue
        dead_in_col = col_data[col_data < dead_threshold]
        frac_dead = len(dead_in_col) / len(col_data)
        if frac_dead >= Config.ANCHOR_DEAD_FRACTION:
            all_dead_vals.extend(dead_in_col.tolist())

    if len(all_dead_vals) >= 2:
        anchor_value = float(np.mean(all_dead_vals))
        return True, anchor_value, len(all_dead_vals)
    elif len(all_dead_vals) == 1:
        if all_dead_vals[0] < 0.20 * control_mean:
            return True, float(all_dead_vals[0]), 1
    return False, np.nan, 0


# ─────────────────────────────────────────────────────────────────────────────
#  6. MODEL OPTIMIZATION (LOO LOOP VIA AIC)
# ─────────────────────────────────────────────────────────────────────────────
def _build_loo_candidate_order(df_valid, residuals_abs, control_mean):
    """Orders candidate points to drop, prioritizing high-variance extremes."""
    df = df_valid.copy()
    if "is_fictional" in df.columns:
        df = df[df["is_fictional"] != True]

    if isinstance(residuals_abs, pd.Series):
        df["residual"] = residuals_abs.reindex(df.index).fillna(0)
    else:
        full_series = pd.Series(residuals_abs, index=df_valid.index)
        df["residual"] = full_series.reindex(df.index).fillna(0)

    high_sd_extreme_cols = set()
    for col in Config.EXTREME_COLS:
        col_vals = df.loc[df["col_idx"] == col, "value"].values
        if len(col_vals) >= 2:
            if np.std(col_vals) > Config.LOO_SD_THRESHOLD * control_mean:
                high_sd_extreme_cols.add(col)

    df["priority"] = df["col_idx"].apply(
        lambda c: 0 if c in high_sd_extreme_cols else 1
    )
    df_sorted = df.sort_values(["priority", "residual"], ascending=[True, False])
    return df_sorted.index.tolist()


def _run_loo_fits(df_valid, init_conc, control_mean, pseudo_zero_conc,
                  fixed_bottom, use_fixed_bottom, skip_dose_gate=False,
                  dilution_factor=None):
    """
    Main model optimization pipeline:
    0. Gate: verify the data actually shows a negative dose-response trend
       before fitting anything, so a flat or inverted dataset never gets a
       reported IC50 or a generated plot.
    1. Perform initial robust weighted fit.
    2. Check for individual experimental outliers via 95% Prediction Band
       to clean bimodal replicate failures (e.g., pipetting drops) before LOO.
    3. Run iterative LOO loop on the cleaned dataset optimized by AIC.

    skip_dose_gate: set by the late-abrupt-drop caller in analyze_drug_pass1,
    which has *already* established a real drop using the mean of the
    highest-dose columns and required >=2 independently dead replicates.
    The gate below instead pools those columns and takes their *median* --
    fragile when one column loses a replicate to MAD outlier removal (a
    near-zero within-column MAD can flag a genuine, tightly-clustered dead
    reading as an outlier), since an uneven column split then skews the
    pooled median toward whichever column kept more points. That already-
    vetted pathway skips this redundant, more fragile re-check rather than
    letting it override a conclusion reached with a more robust test.
    """
    results = []
    df_work = df_valid.copy()
    k_params = 3 if fixed_bottom is not None else 4

    # A global correlation is the wrong test for a plateau-drop-plateau
    # sigmoid: most points sit on the flat plateau (near-zero local
    # correlation), which dilutes even a strong, clean drop in the highest
    # dose columns. Instead, compare the median response at the highest
    # tested doses against the median at the lowest/baseline doses -- the
    # same high/low column convention already used by Config.EXTREME_COLS
    # elsewhere in this pipeline.
    top_anchor_value = None
    if skip_dose_gate:
        drop_frac = None
    else:
        real_pts = df_work[df_work["col_idx"] != -1] if "col_idx" in df_work.columns else df_work
        high_vals = real_pts.loc[real_pts["col_idx"].isin(Config.DOSE_GATE_HIGH_COLS), "value"].values
        low_vals  = real_pts.loc[real_pts["col_idx"].isin(Config.DOSE_GATE_LOW_COLS), "value"].values

        if len(high_vals) >= 2 and len(low_vals) >= 2 and np.median(low_vals) > 0:
            drop_frac = (np.median(low_vals) - np.median(high_vals)) / np.median(low_vals)
        else:
            drop_frac = np.nan

        # Fallback: a potent enough drug/cell-line combination can still be
        # partly active at the lowest tested dose (DOSE_GATE_LOW_COLS), so
        # that comparison understates the real drop -- only tried once the
        # primary check has already failed. Compares the last real dilution
        # point against the true zero-drug control (DOSE_GATE_FALLBACK_REF_COLS);
        # whichever of the two reads *higher* is trusted as the real
        # baseline -- a well reading lower than its neighbor is the more
        # likely one still under residual drug effect (or a one-off
        # artifact), not the other way round.
        if np.isnan(drop_frac) or drop_frac < Config.MIN_DOSE_RESPONSE_DROP:
            ref_a, ref_b = Config.DOSE_GATE_FALLBACK_REF_COLS
            vals_a = real_pts.loc[real_pts["col_idx"] == ref_a, "value"].values
            vals_b = real_pts.loc[real_pts["col_idx"] == ref_b, "value"].values
            if len(vals_a) >= 2 and len(vals_b) >= 2 and len(high_vals) >= 2:
                med_a, med_b = np.median(vals_a), np.median(vals_b)
                fallback_ref = max(med_a, med_b)
                if fallback_ref > 0:
                    fallback_drop = (fallback_ref - np.median(high_vals)) / fallback_ref
                    if fallback_drop >= Config.MIN_DOSE_RESPONSE_DROP:
                        print(f"      [Dose-Response Gate] Primary check failed "
                              f"(drop={drop_frac if not np.isnan(drop_frac) else 'n/a'}), "
                              f"but col {ref_a}={med_a:.4f} vs col {ref_b}={med_b:.4f} shows "
                              f"the lowest tested dose hadn't recovered -- using "
                              f"{fallback_ref:.4f} as the true baseline instead: "
                              f"drop={fallback_drop:.4f}. Proceeding with a fictional Top "
                              f"anchor at {fallback_ref:.4f}.")
                        drop_frac = fallback_drop
                        top_anchor_value = fallback_ref

    if drop_frac is not None and (np.isnan(drop_frac) or drop_frac < Config.MIN_DOSE_RESPONSE_DROP):
        print(f"      [Dose-Response Gate] Highest-dose columns are not meaningfully "
              f"below the baseline plateau (drop={drop_frac if not np.isnan(drop_frac) else 'n/a'}, "
              f"needs >= {Config.MIN_DOSE_RESPONSE_DROP:.0%}). "
              f"Skipping fit -- this dataset will not get an IC50 or a plot.")
        return [{
            "Run": "No dose-response detected (flat/inverted)",
            "Removed_Point": "None",
            "Removed_Point_conc": None,
            "Removed_Point_val": None,
            "IC50_uM": np.nan,
            "HillSlope": np.nan,
            "Bottom": np.nan,
            "Top": np.nan,
            "R2": np.nan,
            "AIC": np.nan,
            "Fixed_Bottom": use_fixed_bottom,
            "Fallback_Pass2": False,
            "N_Removed": 0,
            "Vertical_Purge_Points": [],
            "SE_IC50": np.nan,
            "IC50_CI95_lo": np.nan,
            "IC50_CI95_hi": np.nan,
            "IC50_Boot_lo": np.nan,
            "IC50_Boot_hi": np.nan,
            "IC50_Boot_median": np.nan,
            "Fit_Status": "REJECTED_NO_DOSE_RESPONSE",
            "Skip_Plot": True,
        }]

    if top_anchor_value is not None:
        # Reassign df_valid too (not just df_work) and re-derive df_work
        # from it, so both stay on the same index -- df_work hasn't
        # diverged from df_valid yet at this point (no LOO removal has run
        # yet), so this is equivalent to injecting the anchor before
        # _run_loo_fits was even called, exactly like the existing
        # Bottom-anchor branches in analyze_drug_pass1 already do. Adding
        # points to df_work alone would leave N_Removed's "len(df_valid) -
        # len(df_work)" bookkeeping permanently offset by the anchor count.
        df_top_fict = create_anchor_points(
            df_work, pseudo_zero_conc / (dilution_factor or Config.SERIAL_DILUTION_FACTOR), top_anchor_value)
        df_valid = pd.concat([df_valid, df_top_fict], ignore_index=True)
        df_work = df_valid.copy()

    # ── 1. INITIAL ROBUST FIT ────────────────────────────────────────────────
    # We use a weighted fit right from the start so that high-variance bimodal 
    # columns don't artificially widen the prediction bands.
    popt, r2, residuals, pcov = run_single_fit(
        df_work["concentration"].values, df_work["value"].values,
        init_conc, control_mean, pseudo_zero_conc, fixed_bottom=fixed_bottom
    )
    ic50 = 10**popt[2]
    stats0 = _compute_se_and_ci(popt, pcov, len(df_work))
    initial_ss_res = np.sum(residuals**2)
    initial_aic = compute_aic(initial_ss_res, len(df_work), k_params)

    # ── 2. LOW-DOSE BIOLOGICAL PLATEAU FILTER (single pass, non-recursive) ───
    # Wells at the columns nearest the vehicle control are expected to be
    # alive regardless of curve-fit uncertainty — there is no pharmacological
    # reason for a very low dose to kill cells. A well reading far below the
    # established living-cell plateau in this region is treated as a
    # technical dropout (detachment, pipetting error, excess compound), not
    # as evidence of a real effect.
    #
    # This deliberately does NOT reuse the earlier iterative/fit-based
    # approach: recomputing the reference band from the progressively
    # purged fit created a self-reinforcing loop that narrowed the band
    # round after round, purging real plateau noise even in otherwise clean
    # datasets. Here the reference is estimated ONCE, from a robust median
    # of Config.BIO_PREFILTER_REFERENCE_COLS, and is never recalculated
    # against itself. Only downward deviations at
    # Config.BIO_PREFILTER_CHECK_COLS are flagged — a well reading *higher*
    # than the plateau is not biologically implausible and is left to the
    # LOO/AIC stage. A hard cap (Config.BIO_PREFILTER_MAX_REMOVED) limits
    # how much this filter can remove per dataset, and it abstains entirely
    # if the reference region itself is too noisy to trust (e.g. a compound
    # active even at the lowest tested dose, or generalized technical
    # noise) rather than purging blindly.
    vertical_purge_points = []
    ref_vals = df_work.loc[df_work["col_idx"].isin(Config.BIO_PREFILTER_REFERENCE_COLS), "value"].values
    if len(ref_vals) >= 2:
        ref_median = np.median(ref_vals)
        ref_cv = (np.std(ref_vals) / ref_median) if ref_median > 0 else np.inf

        if ref_median > 0 and ref_cv <= Config.BIO_PREFILTER_MAX_REF_CV:
            drop_threshold = ref_median * Config.BIO_PREFILTER_DROP_FRACTION
            candidates = df_work[
                df_work["col_idx"].isin(Config.BIO_PREFILTER_CHECK_COLS)
                & (df_work["value"] < drop_threshold)
            ]
            if not candidates.empty:
                worst = candidates.sort_values("value").head(Config.BIO_PREFILTER_MAX_REMOVED)
                print(f"      [Bio Plateau Filter] Purged {len(worst)} low-dose dropout(s) "
                      f"(below {Config.BIO_PREFILTER_DROP_FRACTION:.0%} of the "
                      f"{ref_median:.3f} reference plateau).")
                for _, prow in worst.iterrows():
                    vertical_purge_points.append({
                        "concentration": prow["concentration"],
                        "value": prow["value"],
                    })
                df_work = df_work.drop(worst.index).copy()

                popt, r2, residuals, pcov = run_single_fit(
                    df_work["concentration"].values, df_work["value"].values,
                    init_conc, control_mean, pseudo_zero_conc, fixed_bottom=fixed_bottom
                )
                ic50 = 10**popt[2]
                stats0 = _compute_se_and_ci(popt, pcov, len(df_work))
                initial_ss_res = np.sum(residuals**2)
                initial_aic = compute_aic(initial_ss_res, len(df_work), k_params)
        else:
            print(f"      [Bio Plateau Filter] Reference columns too noisy "
                  f"(CV={ref_cv:.2f} > {Config.BIO_PREFILTER_MAX_REF_CV:.2f}) — "
                  f"no reliable plateau to compare against, skipping.")

    results.append({
        "Run": "Initial Cleaned Fit",
        "Removed_Point": "None",
        "Removed_Point_conc": None,
        "Removed_Point_val": None,
        "IC50_uM": ic50,
        "HillSlope": popt[3],
        "Bottom": popt[0],
        "Top": popt[1],
        "R2": r2,
        "AIC": initial_aic,
        "Fixed_Bottom": use_fixed_bottom,
        "Fallback_Pass2": False,
        "N_Removed": len(df_valid) - len(df_work),
        "Vertical_Purge_Points": vertical_purge_points,
        **({"Top_Anchor_Used": True} if top_anchor_value is not None else {}),
        **stats0,
        "IC50_Boot_lo": np.nan,
        "IC50_Boot_hi": np.nan,
        "IC50_Boot_median": np.nan,
        "Fit_Status": "OK",
        "Skip_Plot": False,
    })

    if r2 >= Config.R2_POOR_FIT:
        b_lo, b_hi, b_med = _bootstrap_ic50(df_work, init_conc, control_mean,
                                             pseudo_zero_conc, fixed_bottom)
        results[0]["IC50_Boot_lo"]     = b_lo
        results[0]["IC50_Boot_hi"]     = b_hi
        results[0]["IC50_Boot_median"] = b_med
        return results

    # ── 3. ITERATIVE LOO LOOP (AIC OPTIMIZED) ────────────────────────────────
    eliminated_indices = []
    current_aic  = initial_aic
    current_popt = popt

    for loo_round in range(1, Config.LOO_MAX_REMOVED + 1):
        res_abs = np.abs(
            df_work["value"].values -
            log_4pl(np.log10(df_work["concentration"].values), *current_popt)
        )
        res_abs_series = pd.Series(res_abs, index=df_work.index)
        candidate_order = _build_loo_candidate_order(df_work, res_abs_series, control_mean)

        best_aic  = current_aic
        best_idx  = None
        best_popt = current_popt
        best_pcov = None
        best_r2   = None

        for cand_idx in candidate_order:
            df_try = df_work.drop(cand_idx)
            if len(df_try) < 10:
                continue
            try:
                p, r, res_try, cov = run_single_fit(
                    df_try["concentration"].values, df_try["value"].values,
                    init_conc, control_mean, pseudo_zero_conc,
                    fixed_bottom=fixed_bottom
                )
                ss_try = np.sum(res_try**2)
                aic_try = compute_aic(ss_try, len(df_try), k_params)
                
                if aic_try < best_aic:
                    best_aic  = aic_try
                    best_idx  = cand_idx
                    best_popt = p
                    best_pcov = cov
                    best_r2   = r
            except (RuntimeError, OptimizeWarning):
                continue

        if best_idx is None:
            print(f"      LOO stopped in round {loo_round}: No candidates decrease AIC.")
            break
            
        delta_aic = best_aic - current_aic
        if delta_aic > Config.AIC_MIN_IMPROVEMENT:
            print(f"      LOO stopped in round {loo_round}: AIC decrease ({delta_aic:.2f}) does not justify dropping data.")
            break

        elim_row = df_work.loc[best_idx]
        eliminated_indices.append(best_idx)
        df_work      = df_work.drop(best_idx)
        current_aic  = best_aic
        current_popt = best_popt

        ic50_loo = 10**best_popt[2]
        stats_loo = _compute_se_and_ci(best_popt, best_pcov, len(df_work))
        detail = (f"Col {int(elim_row['col_idx'])} "
                  f"({elim_row['concentration']:.4f} uM), "
                  f"Value: {elim_row['value']:.4f}")
        
        results.append({
            "Run": f"LOO round #{loo_round} ({len(df_valid) - len(df_work)} total pts removed)",
            "Removed_Point": detail,
            "Removed_Point_conc": elim_row["concentration"],
            "Removed_Point_val":  elim_row["value"],
            "IC50_uM":   ic50_loo,
            "HillSlope": best_popt[3],
            "Bottom":    best_popt[0],
            "Top":       best_popt[1],
            "R2":        best_r2,
            "AIC":       best_aic,
            "Fixed_Bottom": use_fixed_bottom,
            "Fallback_Pass2": True if results[0]["N_Removed"] > 0 else False,
            "N_Removed": len(df_valid) - len(df_work),
            **stats_loo,
            "IC50_Boot_lo": np.nan,
            "IC50_Boot_hi": np.nan,
            "IC50_Boot_median": np.nan,
            "Fit_Status": "OK",
            "Skip_Plot": False,
        })

        if loo_round == 2 and current_popt[3] >= 0.0:
            print(f"      [Notice] Slope remains positive ({current_popt[3]:.3f}) after 2 LOO rounds.")
            print("               Aborting LOO loop to preserve data integrity on flat/inactive curves.")
            return [results[0]]

        if best_r2 >= Config.R2_POOR_FIT:
            print(f"      LOO converged in round {loo_round}: "
                  f"R²={best_r2:.3f}, {len(df_valid) - len(df_work)} total pts removed")
            break
    else:
         print(f"      LOO reached maximum of {Config.LOO_MAX_REMOVED} removals.")

    # ── 4. REINSTATEMENT PASS ─────────────────────────────────────────────
    # The LOO loop above is greedy/sequential: a point dropped in an early
    # round is never re-evaluated against the FINAL curve, which may have
    # shifted considerably due to LATER removals. A point that looked
    # anomalous against an intermediate fit can end up sitting right on
    # the converged curve. This single (non-recursive) pass checks each
    # dropped point's residual against the final fit and current residual
    # spread, and restores any that no longer look anomalous.
    if eliminated_indices:
        final_popt = current_popt
        current_resid = np.abs(
            df_work["value"].values -
            log_4pl(np.log10(df_work["concentration"].values), *final_popt)
        )
        resid_sd = np.std(current_resid) if len(current_resid) > 1 else 0.0

        reinstated = []
        if resid_sd > 0:
            for idx in eliminated_indices:
                row = df_valid.loc[idx]
                pred = log_4pl(np.log10(row["concentration"]), *final_popt)
                resid_final = np.abs(row["value"] - pred)
                if resid_final < Config.LOO_REINSTATE_SD * resid_sd:
                    reinstated.append(idx)

        if reinstated:
            print(f"      [Reinstatement] {len(reinstated)} previously dropped point(s) "
                  f"now sit within {Config.LOO_REINSTATE_SD:.1f} SD of the final curve "
                  f"— restoring and refitting.")
            df_work = pd.concat([df_work, df_valid.loc[reinstated]]).sort_index()

            popt_r, r2_r, res_r, pcov_r = run_single_fit(
                df_work["concentration"].values, df_work["value"].values,
                init_conc, control_mean, pseudo_zero_conc, fixed_bottom=fixed_bottom
            )
            ss_r  = np.sum(res_r**2)
            aic_r = compute_aic(ss_r, len(df_work), k_params)
            stats_r = _compute_se_and_ci(popt_r, pcov_r, len(df_work))

            results.append({
                "Run": f"Reinstated ({len(reinstated)} pt restored, "
                       f"{len(df_valid) - len(df_work)} total pts removed)",
                "Removed_Point": "None",
                "Removed_Point_conc": None,
                "Removed_Point_val": None,
                "IC50_uM":   10**popt_r[2],
                "HillSlope": popt_r[3],
                "Bottom":    popt_r[0],
                "Top":       popt_r[1],
                "R2":        r2_r,
                "AIC":       aic_r,
                "Fixed_Bottom": use_fixed_bottom,
                "Fallback_Pass2": results[0].get("Fallback_Pass2", False),
                "N_Removed": len(df_valid) - len(df_work),
                **stats_r,
                "IC50_Boot_lo": np.nan,
                "IC50_Boot_hi": np.nan,
                "IC50_Boot_median": np.nan,
                "Fit_Status": "OK",
                "Skip_Plot": False,
            })
            current_popt = popt_r

    if len(results) > 0:
        b_lo, b_hi, b_med = _bootstrap_ic50(df_work, init_conc, control_mean,
                                             pseudo_zero_conc, fixed_bottom)
        results[-1]["IC50_Boot_lo"]     = b_lo
        results[-1]["IC50_Boot_hi"]     = b_hi
        results[-1]["IC50_Boot_median"] = b_med

    if top_anchor_value is not None:
        for r in results:
            r["Top_Anchor_Used"] = True

    return results

# ─────────────────────────────────────────────────────────────────────────────
#  7. EXPERIMENTAL PIPELINE PASSES
# ─────────────────────────────────────────────────────────────────────────────
def analyze_drug_pass1(df_drug, drug_name, volume, base_conc, file_label,
                       exact_conc=False, sat_drug=None, dilution_factor=None):
    """PASSED 1: Adaptive algorithm utilizing localized data constraints."""
    dilution_factor = dilution_factor or Config.SERIAL_DILUTION_FACTOR
    df_pts, df_valid, control_mean, init_conc, pseudo_zero_conc = \
        _prepare_plate_data(df_drug, base_conc, volume, exact_conc=exact_conc,
                            sat_drug=sat_drug, dilution_factor=dilution_factor)

    X_all = df_valid["concentration"].values
    Y_all = df_valid["value"].values

    results = []
    meta = {
        "needs_fallback": False,
        "late_abrupt_drop": False,
        "r2_free": None,
        "bottom_free": None,
        "top_free": None,
        "bottom_top_ratio": None,
        "control_mean": control_mean,
        "init_conc": init_conc,
        "pseudo_zero_conc": pseudo_zero_conc,
        "df_pts": df_pts,
        "df_valid": df_valid,
        "df_drug": df_drug,
        "volume": volume,
        "base_conc": base_conc,
        "dilution_factor": dilution_factor,
    }

    try:
        popt, r2, _, _ = run_single_fit(X_all, Y_all, init_conc,
                                        control_mean, pseudo_zero_conc)
        meta["r2_free"]    = r2
        meta["bottom_free"] = popt[0]
        meta["top_free"]    = popt[1]
        meta["bottom_top_ratio"] = popt[0] / popt[1] if popt[1] > 0 else None

        late_drop = _is_late_abrupt_drop(df_valid, control_mean)
        meta["late_abrupt_drop"] = late_drop

        should_anchor, anchor_value, n_dead_anchor = _should_add_anchor_point(
            df_valid, control_mean
        )

        if late_drop:
            dead_mask = df_valid["value"] < Config.DEAD_WELL_THRESHOLD * control_mean
            dead_vals = df_valid.loc[dead_mask, "value"].values

            if len(dead_vals) >= 2:
                local_bottom = float(np.mean(dead_vals))
                print(f"    [Pass 1] {drug_name}: Late drop detected. "
                      f"Plateau anchored locally: fixed Bottom = {local_bottom:.4f}")
                
                df_fict = create_anchor_points(df_valid, init_conc * dilution_factor, local_bottom)
                df_valid_fit = pd.concat([df_valid, df_fict], ignore_index=True)

                results = _run_loo_fits(df_valid_fit, init_conc, control_mean,
                                        pseudo_zero_conc,
                                        fixed_bottom=local_bottom,
                                        use_fixed_bottom=True,
                                        skip_dose_gate=True,
                                        dilution_factor=dilution_factor)
                for r in results:
                    r["Fallback_Pass2"] = False
                meta["needs_fallback"] = False
                if popt[1] > 0:
                    meta["bottom_top_ratio"] = local_bottom / popt[1]
                meta["df_valid"] = df_valid_fit
            else:
                if r2 < Config.R2_POOR_FIT:
                    meta["needs_fallback"] = True
                    print(f"    [Pass 1] {drug_name}: R²={r2:.3f} < {Config.R2_POOR_FIT}, "
                          f"late drop but insufficient data points -> queueing Pass 2.")
                else:
                    print(f"    [Pass 1] {drug_name}: R²={r2:.3f} OK -> accepted.")
                    results = _run_loo_fits(df_valid, init_conc, control_mean,
                                            pseudo_zero_conc, fixed_bottom=None,
                                            use_fixed_bottom=False,
                                            dilution_factor=dilution_factor)
        else:
            if r2 >= Config.R2_POOR_FIT:
                reason = f"R²={r2:.3f} OK"
            else:
                reason = "No cytotoxic response curve recognized."
            print(f"    [Pass 1] {drug_name}: {reason} -> fit processed.")

            if should_anchor and r2 < Config.R2_POOR_FIT:
                print(f"    [Well -1] {drug_name}: {n_dead_anchor} replicates "
                      f"clearly dead at high concentrations -> anchoring.")
                df_fict = create_anchor_points(df_valid, init_conc * dilution_factor, anchor_value)
                df_valid_anchor = pd.concat([df_valid, df_fict], ignore_index=True)
                # _should_add_anchor_point checks columns 0 and 1 individually,
                # so it catches a drop confined to just one of them (e.g. col 0
                # fully dead, col 1 unaffected). The Dose-Response Gate instead
                # pools both columns into one combined median, which a
                # single-column drop like that doesn't clear -- diluted by the
                # unaffected column -- and would reject a dataset whose dead
                # replicates we just used to anchor the fit. Skip it here for
                # the same reason the late-abrupt-drop branch above does: the
                # anchor itself is already stronger, per-column evidence of a
                # real dose-response than the gate's pooled check provides.
                results = _run_loo_fits(df_valid_anchor, init_conc, control_mean,
                                        pseudo_zero_conc,
                                        fixed_bottom=anchor_value,
                                        use_fixed_bottom=True,
                                        skip_dose_gate=True,
                                        dilution_factor=dilution_factor)
                for r in results:
                    r["Fallback_Pass2"] = False
                meta["needs_fallback"] = False
                if popt[1] > 0:
                    meta["bottom_top_ratio"] = anchor_value / popt[1]
                meta["df_valid"] = df_valid_anchor
            else:
                results = _run_loo_fits(df_valid, init_conc, control_mean,
                                        pseudo_zero_conc, fixed_bottom=None,
                                        use_fixed_bottom=False,
                                        dilution_factor=dilution_factor)

    except (RuntimeError, ValueError) as e:
        print(f"    [Pass 1] Process exception for {drug_name} in {file_label}: {e}")
        meta["needs_fallback"] = False

    return results, df_pts, df_valid, meta


def analyze_drug_pass2(meta, drug_name, file_label, ref_bottom):
    """PASS 2: Contextual fallback using median parameters of identical cell lines."""
    df_valid = meta["df_valid"]
    init_conc = meta["init_conc"]
    control_mean = meta["control_mean"]
    pseudo_zero_conc = meta["pseudo_zero_conc"]
    dilution_factor = meta.get("dilution_factor") or Config.SERIAL_DILUTION_FACTOR

    print(f"    [Pass 2] {drug_name}: fixed Bottom reference = {ref_bottom:.4f}")

    results = []
    if meta.get("late_abrupt_drop", False):
        df_fict = create_anchor_points(df_valid, init_conc * dilution_factor, ref_bottom)
        df_valid = pd.concat([df_valid, df_fict], ignore_index=True)
        meta["df_valid"] = df_valid

    try:
        results = _run_loo_fits(df_valid, init_conc, control_mean,
                                pseudo_zero_conc, fixed_bottom=ref_bottom,
                                use_fixed_bottom=True,
                                dilution_factor=dilution_factor)
        for r in results:
            r["Fallback_Pass2"] = True
    except (RuntimeError, ValueError) as e:
        print(f"    [Pass 2] Execution error on {drug_name} in {file_label}: {e}")

    return results


_STOCK_SUFFIX_RE = re.compile(r'_2$', re.IGNORECASE)


def _drug_base_name(name):
    """
    Strips a trailing '_2' marker from a drug name for merge-candidate
    comparison only.

    Two stocks of the same drug are normally told apart by Volume (the
    real pipetted µL, see analyze_drug_merged). But when both stocks
    happen to be pipetted at the identical volume, Volume can't serve as
    the distinguishing key without lying about what was physically
    pipetted -- so the second stock is instead registered under its own
    literal name with a '_2' suffix (e.g. "Osimertinib_2"), keeping
    Volume truthful. This strips that suffix so the two blocks are still
    recognized as the same underlying drug for merging; the literal
    suffixed name is left untouched everywhere else (registration
    lookup, per-block plot/result labeling).
    """
    return _STOCK_SUFFIX_RE.sub('', name.strip())


def analyze_drug_merged(entry1, entry2):
    """
    Combines two blocks that share the same cell line and drug name but
    were tested at different starting concentrations (e.g. more stock
    volume pipetted from the same stock solution) into a single fit
    spanning their combined concentration range.

    Only called (see the caller in the pipeline entrypoint) once both
    individual fits have already independently demonstrated a real
    dose-response of their own -- this reuses their already MAD-cleaned,
    already-anchored-if-needed data (meta["df_valid"]) rather than
    re-deriving QC from scratch on the merged set, and skips the
    dose-response gate for the same reason (Config.EXTREME_COLS /
    DOSE_GATE_*/BIO_PREFILTER_* column-index heuristics are also left to
    operate on each block's own columns rather than being redefined for
    a combined, variable-width layout -- a secondary refinement, not
    something the fit's correctness depends on).

    If either half needed a fixed Bottom anchor, the merged fit reuses it
    (averaging if both anchored) rather than re-discovering one -- more
    data across a wider range is exactly what a free fit needs to resolve
    Bottom on its own when a single narrow block could not, so forcing an
    anchor here would throw away exactly the benefit of merging.
    """
    meta1, meta2 = entry1["meta"], entry2["meta"]
    df1 = meta1["df_valid"].copy()
    df2 = meta2["df_valid"].copy()

    # col_idx only has meaning *within* a single block's own concentration
    # series -- block2's would otherwise collide with block1's and confuse
    # the column-based LOO-priority/bio-prefilter heuristics. Offsetting
    # (but preserving the col_idx=-1 fictional-anchor-point sentinel)
    # keeps the merge unambiguous without touching the fit itself, which
    # only cares about each point's own (concentration, value) pair.
    df2["col_idx"] = df2["col_idx"].apply(lambda c: c if c == -1 else c + 1000)

    df_merged = pd.concat([df1, df2], ignore_index=True)

    best1 = max(entry1["results"], key=lambda r: r["R2"]) if entry1["results"] else None
    best2 = max(entry2["results"], key=lambda r: r["R2"]) if entry2["results"] else None
    anchors = [r["Bottom"] for r in (best1, best2) if r and r.get("Fixed_Bottom")]
    fixed_bottom = float(np.mean(anchors)) if anchors else None

    init_conc = max(meta1["init_conc"], meta2["init_conc"])
    control_mean = float(np.mean([meta1["control_mean"], meta2["control_mean"]]))
    pseudo_zero_conc = min(meta1["pseudo_zero_conc"], meta2["pseudo_zero_conc"])

    print(f"    [Combined] {entry1['drug']['name']}: merging {entry1['drug']['volume']} uL "
          f"and {entry2['drug']['volume']} uL blocks ({len(df_merged)} points, "
          f"{df_merged['concentration'].min():.4g}-{df_merged['concentration'].max():.4g} uM)"
          + (f", fixed Bottom = {fixed_bottom:.4f}" if fixed_bottom is not None else ", free fit"))

    try:
        results = _run_loo_fits(df_merged, init_conc, control_mean, pseudo_zero_conc,
                                fixed_bottom=fixed_bottom,
                                use_fixed_bottom=fixed_bottom is not None,
                                skip_dose_gate=True,
                                dilution_factor=max(meta1.get("dilution_factor") or 0,
                                                    meta2.get("dilution_factor") or 0) or None)
    except (RuntimeError, ValueError) as e:
        print(f"    [Combined] Execution error merging {entry1['drug']['name']}: {e}")
        return [], df_merged

    for r in results:
        r["Fallback_Pass2"] = False
    return results, df_merged


def analyze_drug_dilution_corrected(conc_entry, dil_entry, dilution_factor):
    """
    Recovers real values for detector-saturated wells in a block using a
    second reading of the identical physical plate layout at a known
    dilution (a "_DilN" file, see _extract_dilution_factor), instead of
    leaving those wells at the generic ceiling_value that
    extract_plate_grid imputes for every saturated well by default.

    Absorbance vs. concentration is not guaranteed linear across the
    reader's full range (Beer-Lambert holds only at low absorbance), and
    that non-linearity is worst exactly near saturation -- so the nominal
    dilution factor (e.g. 4x) is not trusted to rescale the diluted
    reading back onto the concentrated file's scale. Instead the
    relationship concentrated ~= f(diluted) is fit empirically, using
    only wells where BOTH readings are reliable (neither is saturated):
    a same-instrument mapping between the two readings' own scales,
    valid regardless of whether the underlying absorbance response is
    itself linear.

    Only concentrated-block wells that are saturated AND have a matching,
    non-saturated diluted-block reading at the identical (row_idx,
    col_idx) -- same physical well, per this lab's confirmed identical
    plate layout -- get corrected; anything else keeps its original
    ceiling_value, still flagged is_saturated. Returns None (no third fit
    produced) when there is nothing to correct, or not enough reliable
    overlap between the two files to trust an empirical rescale.
    """
    df_conc = conc_entry["df_valid"].copy()
    df_dil = dil_entry["df_valid"]

    dil_lookup = {
        (row.row_idx, row.col_idx): (row.value, bool(row.is_saturated))
        for row in df_dil.itertuples()
    }

    ref_conc_vals, ref_dil_vals = [], []
    for row in df_conc.itertuples():
        if row.is_saturated:
            continue
        match = dil_lookup.get((row.row_idx, row.col_idx))
        if match is None or match[1] or match[0] == 0:
            continue
        ref_conc_vals.append(row.value)
        ref_dil_vals.append(match[0])

    if len(ref_conc_vals) < 2:
        print(f"    [Dil-Correction] {conc_entry['drug']['name']}: only "
              f"{len(ref_conc_vals)} reliable overlapping well(s) between "
              f"'{conc_entry['file_name']}' and '{dil_entry['file_name']}' -- "
              f"not enough to trust an empirical rescale. Skipped.")
        return None

    ref_conc_vals = np.array(ref_conc_vals)
    ref_dil_vals = np.array(ref_dil_vals)
    if len(ref_conc_vals) >= 4:
        slope, intercept = np.polyfit(ref_dil_vals, ref_conc_vals, 1)
        rescale = lambda v: slope * v + intercept
        rescale_desc = (f"linear fit (slope={slope:.4f}, intercept={intercept:.4f}) "
                        f"over {len(ref_conc_vals)} wells")
    else:
        ratio = float(np.median(ref_conc_vals / ref_dil_vals))
        rescale = lambda v: ratio * v
        rescale_desc = f"median ratio ({ratio:.4f}) over {len(ref_conc_vals)} wells"

    # The rescale is fit only on the diluted values actually observed
    # among the reference wells -- applying it *outside* that range is
    # extrapolation, and extrapolation is exactly where the "absorbance
    # isn't linear near saturation" concern this correction exists to
    # respect bites hardest. A well whose diluted counterpart falls
    # outside [dil_ref_lo, dil_ref_hi] is left uncorrected (still the
    # original ceiling_value, still flagged is_saturated) rather than
    # trusting the line beyond where it was actually measured.
    dil_ref_lo, dil_ref_hi = float(np.min(ref_dil_vals)), float(np.max(ref_dil_vals))

    n_corrected = 0
    n_out_of_range = 0
    for idx in df_conc.index[df_conc["is_saturated"]]:
        row_idx = df_conc.at[idx, "row_idx"]
        col_idx = df_conc.at[idx, "col_idx"]
        match = dil_lookup.get((row_idx, col_idx))
        if match is None or match[1]:
            continue
        dil_val = match[0]
        if not (dil_ref_lo <= dil_val <= dil_ref_hi):
            n_out_of_range += 1
            continue
        df_conc.at[idx, "value"] = rescale(dil_val)
        df_conc.at[idx, "is_saturated"] = False
        n_corrected += 1

    if n_corrected == 0:
        reason = (f"the {n_out_of_range} candidate well(s) found all fell outside the "
                   f"rescale's reliable range [{dil_ref_lo:.4g}, {dil_ref_hi:.4g}] -- "
                   f"correcting them would mean extrapolating"
                   if n_out_of_range > 0 else
                   "no saturated well had a usable (non-saturated) match")
        print(f"    [Dil-Correction] {conc_entry['drug']['name']}: {reason} in "
              f"'{dil_entry['file_name']}'. Nothing to correct.")
        return None

    print(f"    [Dil-Correction] {conc_entry['drug']['name']}: recovered {n_corrected} "
          f"saturated well(s) in '{conc_entry['file_name']}' using {rescale_desc} from "
          f"'{dil_entry['file_name']}' (nominal {dilution_factor}x dilution)"
          + (f"; {n_out_of_range} more stayed uncorrected (outside the reliable "
             f"interpolation range)." if n_out_of_range > 0 else "."))

    best_conc = max(conc_entry["results"], key=lambda r: r["R2"]) if conc_entry["results"] else None
    fixed_bottom = best_conc["Bottom"] if (best_conc and best_conc.get("Fixed_Bottom")) else None

    meta = conc_entry["meta"]
    try:
        results = _run_loo_fits(df_conc, meta["init_conc"], meta["control_mean"],
                                meta["pseudo_zero_conc"], fixed_bottom=fixed_bottom,
                                use_fixed_bottom=fixed_bottom is not None,
                                dilution_factor=meta.get("dilution_factor"))
    except (RuntimeError, ValueError) as e:
        print(f"    [Dil-Correction] Execution error fitting corrected data for "
              f"{conc_entry['drug']['name']}: {e}")
        return None

    if not results:
        return None

    ic50_raw = best_conc["IC50_uM"] if best_conc else None
    best_corrected = max(results, key=lambda r: r["R2"])
    ic50_delta = (best_corrected["IC50_uM"] - ic50_raw
                  if (ic50_raw is not None and best_corrected.get("IC50_uM") is not None)
                  else None)

    for r in results:
        r["Fallback_Pass2"] = False
        r["Saturation_Corrected"] = True
        r["N_Wells_Corrected"] = n_corrected
        r["IC50_Delta_vs_Raw_uM"] = ic50_delta

    if ic50_delta is not None:
        print(f"    [Dil-Correction] IC50 shift after correction: "
              f"{ic50_raw:.4g} -> {best_corrected['IC50_uM']:.4g} uM "
              f"(delta={ic50_delta:+.4g} uM).")

    return results, df_conc


# ─────────────────────────────────────────────────────────────────────────────
#  8. GRAPHICAL EXPORT PANEL
# ─────────────────────────────────────────────────────────────────────────────
def _write_not_reached_note(cell_line, drug_name, volume, file_name, folder_path, reason):
    """
    Writes a small text file next to the plots explaining why a block
    that would normally get one has none. With two expected plots per
    plate file, a missing one is otherwise indistinguishable from a
    silent failure -- this turns it into a documented, expected outcome.
    """
    safe_cell = re.sub(r"[^\w\-]", "_", cell_line)
    safe_drug = re.sub(r"[^\w\-]", "_", drug_name)
    file_base = os.path.splitext(file_name)[0]
    note_path = os.path.join(
        folder_path, f"NotReached_{safe_cell}_{safe_drug}_vol{volume}_{file_base}.txt"
    )
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("No plot was generated for this block.\n\n")
        f.write(f"File: {file_name}\n")
        f.write(f"Cell line: {cell_line}\n")
        f.write(f"Drug: {drug_name}\n")
        f.write(f"Reason: {reason}\n")
    print(f"    -> Note written: {os.path.basename(note_path)}")


def save_single_plot(res, valid_df, drug_name, volume, cell_line, file_name, folder_path):
    """Generates standardized, high-resolution scientific plots."""
    if not res:
        _write_not_reached_note(
            cell_line, drug_name, volume, file_name, folder_path,
            "Processing error during fitting -- no result was produced. "
            "See the console log for the exception raised while analyzing this block."
        )
        return

    run0         = res[0]

    if run0.get("Skip_Plot", False):
        reason = f"{run0.get('Run', 'No dose-response detected')} " \
                 f"(Fit_Status={run0.get('Fit_Status', 'unknown')}). " \
                 f"The highest tested concentration did not show a meaningful drop " \
                 f"below the baseline plateau, so no IC50 could be reported."
        print(f"    -> No plot generated for {drug_name} ({volume} uL): "
              f"{run0.get('Fit_Status', 'no dose-response detected')}.")
        _write_not_reached_note(cell_line, drug_name, volume, file_name, folder_path, reason)
        return

    best_run     = max(res, key=lambda r: r["R2"])
    bottom_fijo  = run0.get("Fixed_Bottom", False)
    fallback_p2  = run0.get("Fallback_Pass2", False)
    n_loo_rounds = len(res) - 1

    elim_colors = plt.cm.Oranges(np.linspace(0.45, 0.95, max(n_loo_rounds, 1)))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('ggplot')

    real_mask = (valid_df["col_idx"] != -1) if "col_idx" in valid_df.columns else pd.Series(True, index=valid_df.index)
    real_df = valid_df[real_mask]
    fict_mask = (valid_df["col_idx"] == -1) if "col_idx" in valid_df.columns else pd.Series(False, index=valid_df.index)
    fict_df = valid_df[fict_mask]

    # Detector-saturated wells were imputed at a ceiling value before
    # fitting -- shown with their own marker so a reader can see which
    # points are censored readings rather than direct measurements.
    if "is_saturated" in real_df.columns:
        sat_plot_mask = real_df["is_saturated"].astype(bool)
    else:
        sat_plot_mask = pd.Series(False, index=real_df.index)
    sat_df = real_df[sat_plot_mask]
    real_df = real_df[~sat_plot_mask]

    # Points purged by the iterative vertical prediction-band filter
    # (pre-LOO) should not be shown as ordinary valid points — they were
    # excluded from the fit used to draw both curves below.
    vp_points = run0.get("Vertical_Purge_Points", [])
    if vp_points:
        vp_set = {(round(p["concentration"], 10), round(p["value"], 10)) for p in vp_points}
        real_df_key = list(zip(real_df["concentration"].round(10), real_df["value"].round(10)))
        keep_mask = [k not in vp_set for k in real_df_key]
        real_df = real_df[keep_mask]

    # Config.X_LIM_MIN/MAX are a sensible default for a "typical" low-uM
    # compound, but a fixed upper bound silently clips the plot -- not the
    # fit itself, curve_fit always sees the real data -- for any compound
    # tested at higher concentrations (e.g. a 30 mM Well 1): the true
    # transition can sit well past 1500 uM, making a perfectly valid fit
    # look like a flat, meaningless line. Expand (never shrink) the axis
    # to always cover the concentrations actually present in this block.
    conc_min = valid_df["concentration"].min()
    conc_max = valid_df["concentration"].max()
    x_lim_min = min(Config.X_LIM_MIN, 10 ** np.floor(np.log10(conc_min)))
    x_lim_max = max(Config.X_LIM_MAX, 10 ** (np.ceil(np.log10(conc_max)) + 0.3))

    ax.set_xscale('log')
    ax.set_xlim(x_lim_min, x_lim_max)

    x_c = np.linspace(np.log10(x_lim_min), np.log10(x_lim_max), 500)

    ax.scatter(real_df["concentration"], real_df["value"],
               color='#2ec4b6', alpha=0.55, s=32, zorder=2, label='Valid points')

    if len(sat_df) > 0:
        ax.scatter(sat_df["concentration"], sat_df["value"],
                   color='#9d4edd', alpha=0.85, s=55, marker='^', zorder=4,
                   linewidths=0.6, edgecolors='#3c096c',
                   label=f'Saturated, imputed ceiling ({len(sat_df)} pt., val={sat_df["value"].iloc[0]:.3f})')

    if vp_points:
        vp_conc = [p["concentration"] for p in vp_points]
        vp_val  = [p["value"] for p in vp_points]
        ax.scatter(vp_conc, vp_val, color='#8d99ae', marker='P', s=90, zorder=5,
                   linewidths=0.8, edgecolors='#2b2d42',
                   label=f'Purged pre-fit ({len(vp_points)} pt., implausible/bimodal)')

    if len(fict_df) > 0:
        fict_conc_val = fict_df["concentration"].iloc[0]
        fict_y_val = fict_df["value"].iloc[0]
        ax.scatter(fict_df["concentration"], fict_df["value"],
                   facecolors='none', edgecolors='#e63946', marker='D',
                   s=80, zorder=5, linewidths=2.0,
                   label=f'Well −1 anchor ({fict_conc_val:.0f} µM, val={fict_y_val:.3f})')

    for i, run in enumerate(res[1:], start=1):
        ec = run.get("Removed_Point_conc")
        ev = run.get("Removed_Point_val")
        if ec is not None:
            ax.scatter([ec], [ev], color=elim_colors[i-1], marker='X',
                       s=110, zorder=6, linewidths=1.0,
                       label=f'Rem. round {i} ({ec:.3g} uM, val={ev:.3f})')

    popt0 = [run0["Bottom"], run0["Top"], np.log10(run0["IC50_uM"]), run0["HillSlope"]]
    ax.plot(10**x_c, log_4pl(x_c, *popt0), color='#0f9f90', linewidth=2.5,
            label=f'Initial Fit  IC50={run0["IC50_uM"]:.2f} µM  R²={run0["R2"]:.3f}')

    if n_loo_rounds > 0 and best_run is not run0:
        popt_best = [best_run["Bottom"], best_run["Top"],
                     np.log10(best_run["IC50_uM"]), best_run["HillSlope"]]
        n_elim = best_run["N_Removed"]
        ax.plot(10**x_c, log_4pl(x_c, *popt_best),
                color='#e63946', linewidth=2.2, linestyle='--',
                label=(f'Best LOO ({n_elim} pts rem.)  '
                       f'IC50={best_run["IC50_uM"]:.2f} µM  '
                       f'R²={best_run["R2"]:.3f}'))

    ax.set_xlabel('Concentration (µM)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Signal / Viability',  fontsize=9, fontweight='bold')

    title_parts = [f"{cell_line}  —  {drug_name} ({volume} µL)"]
    if bottom_fijo:
        title_parts.append(f"[Fixed Bottom = {run0['Bottom']:.3f}]")
    if fallback_p2:
        title_parts.append("[Pass 2 Fallback]")
    ax.set_title("  ".join(title_parts), fontsize=10, fontweight='bold')
    fig.suptitle(f"Robustness Analysis: {file_name}", fontsize=8, color='#555555')

    ax.legend(fontsize=7, loc='lower left', framealpha=0.85)
    plt.tight_layout()

    safe_cell = re.sub(r"[^\w\-]", "_", cell_line)
    safe_drug = re.sub(r"[^\w\-]", "_", drug_name)
    file_base = os.path.splitext(file_name)[0]
    plot_out  = os.path.join(folder_path,
                             f"Plot_{safe_cell}_{safe_drug}_vol{volume}_{file_base}.png")
    plt.savefig(plot_out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    → Plot saved: {os.path.basename(plot_out)}")


# ─────────────────────────────────────────────────────────────────────────────
#  9. PIPELINE EXECUTION ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    folder_path = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "!"*70)
    print("  [IMPORTANT NOTICE] EXPERIMENTAL DESIGN REQUIREMENT:")
    print("  The highest concentration of your dilution series MUST be in Well 1.")
    print("  The script assumes a continuous serial dilution starting from Column 1")
    print(f"  (Well 1) to Column 11 (factor per drug from Starting_Concentrations.xlsx,")
    print(f"  {Config.SERIAL_DILUTION_FACTOR:g}-fold where the dilution column is empty or missing).")
    print("  Do NOT leave Well 1 empty or start the series")
    print("  in Well 2, as this will shift and invalidate all calculated IC50 values.")
    print("!"*70 + "\n")

    print("Loading starting concentrations...")
    conc_file = os.path.join(folder_path, "Starting_Concentrations.xlsx")
    if not os.path.exists(conc_file):
        candidates = glob.glob(os.path.join(folder_path, "Starting_*.xlsx")) + \
                     glob.glob(os.path.join(folder_path, "Concentra*.xlsx"))
        if candidates:
            conc_file = candidates[0]
        else:
            raise FileNotFoundError(
                f"Could not find 'Starting_Concentrations.xlsx' in {folder_path}"
            )

    df_conc = pd.read_excel(conc_file)
    conc_dict = {}
    dilution_dict = {}

    # Detect layout configuration
    _has_volume_col = len(df_conc.columns) >= 3
    _has_dilution_col = len(df_conc.columns) >= 4
    if _has_dilution_col:
        print("  [Starting Concentrations] 4-column format verified: "
              "(Drug, Volume, Concentration, Dilution factor)")
    elif _has_volume_col:
        print("  [Starting Concentrations] Standard 3-column format verified: (Drug, Volume, Concentration)")
        print(f"  No dilution column: every drug uses {Config.SERIAL_DILUTION_FACTOR:g}-fold dilutions.")
    else:
        print("\n" + "!"*70)
        print("  [WARNING] Expected format not detected in Starting_Concentrations.xlsx!")
        print("  Please ensure your spreadsheet includes all 3 specified columns:")
        print("  -> [Drug], [Volume], and [Concentration] (Well 1)")
        print("  ")
        print("  Proceeding with default assumptions (scaling based on 2 µL reference volume)...")
        print("!"*70 + "\n")

    for idx, row in df_conc.iterrows():
        name = str(row.iloc[0]).strip().lower()
        if _has_volume_col:
            vol  = int(row.iloc[1])
            val  = float(row.iloc[2])
            if name not in conc_dict:
                conc_dict[name] = {}
            conc_dict[name][vol] = val
            if _has_dilution_col:
                factor = _parse_dilution_factor(row.iloc[3], f"{name} ({vol} uL)")
                if factor is not None:
                    dilution_dict.setdefault(name, {})[vol] = factor
        else:
            val = float(row.iloc[1])
            conc_dict[name] = val

    xls_files = glob.glob(os.path.join(folder_path, "*.xls"))
    xls_files = [f for f in xls_files if not os.path.basename(f).startswith("~$")]
    print(f"Found {len(xls_files)} plate files to analyze.")

    skipped_files = []
    pass1_results = []
    ref_bottoms = {}
    unregistered_drug_warnings = []
    merge_candidates = []

    # ── PASS 1 ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PASS 1: Adaptive, free fit for all datasets")
    print("="*60)

    for file in xls_files:
        file_name = os.path.basename(file)
        print(f"\nProcessing: {file_name}")

        cell_lines, drugs = parse_filename(file_name, conc_dict, dilution_dict)
        if cell_lines is not None:
            for d in drugs:
                d["dilution_factor"] = d.get("dilution_factor") or Config.SERIAL_DILUTION_FACTOR
        if cell_lines is None:
            print(f"  [SKIPPED] Could not identify cell line(s) or a registered drug in the "
                  f"filename. Format must be '<cell_lines>; <drugs>.xls' (';' or ',' as separator).")
            skipped_files.append(file_name)
            continue

        try:
            df_clean, sat_clean, ceiling_value, n_saturated = extract_plate_grid(file)
        except ValueError as e:
            print(f"  [WARNING] Could not locate a valid plate grid ({e}). Skipping.")
            skipped_files.append(file_name)
            continue

        if n_saturated > 0:
            print(f"  [Saturation] {n_saturated} well(s) flagged as detector-saturated "
                  f"-> imputed as {ceiling_value:.4f} (highest reliable reading in this file).")

        print(f"  Lines: {', '.join(cell_lines)}")
        for i, d in enumerate(drugs, 1):
            if d.get("is_no_drug", False):
                print(f"  Drug {i}: {d['name']} (no compound applied -- block will be skipped)")
            else:
                print(f"  Drug {i}: {d['name']} ({d['volume']} uL, base {d['base_conc']} uM, "
                      f"1:{d['dilution_factor']:g} serial dilution)")

        assignments = build_assignments(cell_lines, drugs, df_clean, sat_clean)
        file_block_entries = []

        for cell_line, drug, df_rows, sat_rows in assignments:
            print(f"  → {cell_line} / {drug['name']}")
            if cell_line.strip().lower() in Config.NO_CELL_LINE_TOKENS:
                print(f"    [Skipped] '{cell_line}' marks wells with no cells "
                      f"(blank/background control) -- no dose-response to fit, "
                      f"no output row or plot generated for this block.")
                _write_not_reached_note(
                    cell_line, drug['name'], drug['volume'], file_name, folder_path,
                    f"'{cell_line}' marks a blank/background control block (no "
                    f"cells present) -- there is no dose-response to fit."
                )
                continue
            if drug.get("is_no_drug", False):
                print(f"    [Skipped] '{drug['name']}' marks wells with no compound "
                      f"applied -- no dose-response to fit, no output row or plot "
                      f"generated for this block.")
                _write_not_reached_note(
                    cell_line, drug['name'], drug['volume'], file_name, folder_path,
                    "No compound was applied to these wells (marked as a no-drug "
                    "block in the filename) -- there is no dose-response to fit."
                )
                continue
            if drug.get("is_unregistered_placeholder", False):
                print(f"    [Skipped] '{drug['name']}' does not match any drug "
                      f"registered in Starting_Concentrations.xlsx -- refusing to fit "
                      f"with a fabricated concentration. Likely a spelling/naming "
                      f"mismatch (e.g. this file says '{drug['name']}' but the "
                      f"spreadsheet has it under a different name).")
                _write_not_reached_note(
                    cell_line, drug['name'], drug['volume'], file_name, folder_path,
                    f"'{drug['name']}' does not match any drug registered in "
                    f"Starting_Concentrations.xlsx. No fit was attempted -- fitting "
                    f"with a fabricated concentration would produce a misleading "
                    f"IC50. Add this drug to Starting_Concentrations.xlsx and re-run "
                    f"-- check for a spelling/naming mismatch (e.g. 'Gemcitabina' vs "
                    f"'Gemcitabine'), or if this is a drug combination, register it "
                    f"as its own row (e.g. '{drug['name']}')."
                )
                unregistered_drug_warnings.append(
                    f"{file_name} / {cell_line} / '{drug['name']}' -> NOT fitted "
                    f"(no matching row in Starting_Concentrations.xlsx)"
                )
                continue
            res, df_pts, df_valid, meta = analyze_drug_pass1(
                df_rows, drug['name'], drug['volume'], drug['base_conc'], file_name,
                exact_conc=drug.get('exact_conc', False), sat_drug=sat_rows,
                dilution_factor=drug['dilution_factor']
            )

            entry = {
                "file_name": file_name,
                "cell_line": cell_line,
                "drug": drug,
                "df_pts": df_pts,
                "df_valid": df_valid,
                "meta": meta,
                "results": res,
            }
            pass1_results.append(entry)
            file_block_entries.append(entry)

            # Accumulate ratio using the Median for robust fallback statistics
            if res and meta["r2_free"] is not None:
                if (meta["r2_free"] >= Config.R2_POOR_FIT
                        and meta["bottom_top_ratio"] is not None):
                    ref_bottoms.setdefault(cell_line, []).append(
                        meta["bottom_top_ratio"]
                    )
                    print(f"    → Bottom/Top Ratio={meta['bottom_top_ratio']:.4f} added for '{cell_line}'")

        # A file naming the same drug twice under the same cell line (e.g.
        # more stock volume pipetted from the same stock solution, to reach
        # a higher top concentration) produces two separate blocks above.
        # Structural detection only here -- whether either block actually
        # needs Pass 2 isn't known yet, so the eligibility/fit check (and
        # the merge itself) is deferred until after Pass 2 resolves both
        # blocks' final results (see below).
        # Same base drug (ignoring an optional '_2' second-stock suffix,
        # see _drug_base_name) but not registered identically -- either a
        # different Volume (two stocks of different physical volumes) or a
        # different literal name via the '_2' suffix (two stocks pipetted
        # at the same volume, distinguished by name instead) -- so these
        # are genuinely two stocks of one drug, not an accidental duplicate
        # block.
        if len(file_block_entries) == 2 and len(cell_lines) == 1 and len(drugs) >= 2:
            same_registration = (drugs[0]["name"].strip().lower() == drugs[1]["name"].strip().lower()
                                  and drugs[0].get("volume") == drugs[1].get("volume"))
            if (_drug_base_name(drugs[0]["name"]).lower() == _drug_base_name(drugs[1]["name"]).lower()
                    and not same_registration):
                merge_candidates.append((file_block_entries[0], file_block_entries[1]))

    # ── PASS 2 ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PASS 2: Contextual reference bottom-fitting fallback")
    print("="*60)

    for entry in pass1_results:
        if not entry["meta"]["needs_fallback"]:
            continue

        cell_line = entry["cell_line"]
        drug = entry["drug"]
        file_name = entry["file_name"]

        if cell_line in ref_bottoms and ref_bottoms[cell_line]:
            ref_ratio   = np.median(ref_bottoms[cell_line])
            top_actual  = entry["meta"]["control_mean"]
            ref_bottom  = ref_ratio * top_actual
            print(f"\n  {cell_line} / {drug['name']} ({file_name})")
            print(f"    Median reference ratio={ref_ratio:.4f}, Actual Top={top_actual:.4f} → Fixed Bottom={ref_bottom:.4f}")
            res2 = analyze_drug_pass2(entry["meta"], drug['name'], file_name, ref_bottom)
            entry["results"] = res2
        else:
            print(f"\n  [No Reference] {cell_line} / {drug['name']} ({file_name}): No baseline available.")
            meta = entry["meta"]
            try:
                res_fallback = _run_loo_fits(
                    meta["df_valid"], meta["init_conc"], meta["control_mean"],
                    meta["pseudo_zero_conc"], fixed_bottom=None, use_fixed_bottom=False,
                    dilution_factor=meta.get("dilution_factor")
                )
                for r in res_fallback:
                    r["Fallback_Pass2"] = False
                    r["Fixed_Bottom"] = False
                entry["results"] = res_fallback
            except (RuntimeError, ValueError) as e:
                print(f"    Error in fallback free fit: {e}")

    # ── Dilution-pair saturation correction ─────────────────────────────────
    # Groups blocks (same cell line + drug) across a concentrated file and
    # its "_DilN" re-read of the identical physical plate (see
    # analyze_drug_dilution_corrected). Snapshot taken now, before the merge
    # section below appends its own combined rows -- those have no diluted
    # counterpart of their own, so leaving them out of this grouping avoids
    # pointless (if harmless) lookups against them.
    dilution_groups = {}
    for entry in pass1_results:
        factor = _extract_dilution_factor(entry["file_name"])
        group_key = (
            _dilution_pair_key(entry["file_name"]).lower(),
            entry["cell_line"].strip().lower(),
            entry["drug"]["name"].strip().lower(),
        )
        bucket = dilution_groups.setdefault(group_key, {"concentrated": [], "diluted": []})
        if factor is None:
            bucket["concentrated"].append(entry)
        else:
            bucket["diluted"].append((entry, factor))

    dilution_pairs_found = [
        (bucket["concentrated"][0], bucket["diluted"][0][0], bucket["diluted"][0][1])
        for bucket in dilution_groups.values()
        if bucket["concentrated"] and bucket["diluted"]
    ]
    if dilution_pairs_found:
        print("\n" + "="*60)
        print("Dilution-pair saturation correction")
        print("="*60)
    for conc_entry, dil_entry, factor in dilution_pairs_found:
        if not (conc_entry["results"] and conc_entry["results"][0].get("Fit_Status") == "OK"):
            continue
        corrected = analyze_drug_dilution_corrected(conc_entry, dil_entry, factor)
        if corrected is None:
            continue
        corrected_results, df_corrected = corrected
        corrected_entry = {
            "file_name": conc_entry["file_name"],
            "cell_line": conc_entry["cell_line"],
            "drug": {**conc_entry["drug"], "name": f"{conc_entry['drug']['name']} (Dil-Corrected)"},
            "df_pts": None,
            "df_valid": df_corrected,
            "meta": {"needs_fallback": False},
            "results": corrected_results,
        }
        pass1_results.append(corrected_entry)

    # ── Same-drug, multiple-concentration merge ────────────────────────────
    # Now that Pass 2 has resolved every block's final results (including
    # ones that needed a fallback fit), add a third, combined fit for any
    # pair that both independently show a real dose-response -- more data
    # across their full combined concentration range than either block
    # alone, with no additional confound as long as both were run under
    # the same conditions (see analyze_drug_merged).
    if merge_candidates:
        print("\n" + "="*60)
        print("Same-drug, multiple-concentration merges")
        print("="*60)
    for entry1, entry2 in merge_candidates:
        if not (entry1["results"] and entry2["results"]
                and entry1["results"][0].get("Fit_Status") == "OK"
                and entry2["results"][0].get("Fit_Status") == "OK"):
            continue
        merged_results, df_merged = analyze_drug_merged(entry1, entry2)
        if not merged_results:
            continue
        merged_init_conc = max(entry1["meta"]["init_conc"], entry2["meta"]["init_conc"])
        merged_entry = {
            "file_name": entry1["file_name"],
            "cell_line": entry1["cell_line"],
            "drug": {
                **entry1["drug"],
                "name": f"{_drug_base_name(entry1['drug']['name'])} (Combined)",
                "volume": f"{entry1['drug']['volume']}+{entry2['drug']['volume']}",
                # base_conc/exact_conc are set so the consolidation step's
                # Well1_Concentration_uM formula (which multiplies by
                # volume/2.0 when exact_conc is False) doesn't try to do
                # arithmetic on the "2+10"-style combined volume label above.
                "base_conc": merged_init_conc,
                "exact_conc": True,
            },
            "df_pts": None,
            "df_valid": df_merged,
            "meta": {"needs_fallback": False},
            "results": merged_results,
        }
        pass1_results.append(merged_entry)

    # ── CONSOLIDATE RESULTS ──────────────────────────────────────────────────
    all_runs_data = []
    for entry in pass1_results:
        for r in entry["results"]:
            r.update({
                "File": entry["file_name"],
                "Cell_Line": entry["cell_line"],
                "Drug": entry["drug"]["name"],
                "Stock_Volume_uL": entry["drug"]["volume"],
                "Serial_Dilution_Factor": entry["drug"].get("dilution_factor"),
                "Well1_Concentration_uM": entry["drug"]["base_conc"] if entry["drug"].get("exact_conc", False) else (entry["drug"]["base_conc"] * (entry["drug"]["volume"] / 2.0))
            })
            all_runs_data.append(r)

        save_single_plot(entry["results"], entry["df_valid"],
                         entry["drug"]["name"], entry["drug"]["volume"],
                         entry["cell_line"], entry["file_name"], folder_path)

    if skipped_files:
        print(f"\n[WARNING] {len(skipped_files)} file(s) skipped:")
        for f in skipped_files:
            print(f"  - {f}")

    df_final = pd.DataFrame(all_runs_data)
    cols = [
        "File", "Cell_Line", "Drug", "Stock_Volume_uL",
        "Well1_Concentration_uM", "Serial_Dilution_Factor", "Fit_Status", "Run", "Removed_Point",
        "IC50_uM", "IC50_CI95_lo", "IC50_CI95_hi",
        "IC50_Boot_lo", "IC50_Boot_hi", "IC50_Boot_median",
        "SE_IC50_uM",
        "HillSlope", "SE_HillSlope", "Hill_CI95_lo", "Hill_CI95_hi",
        "Bottom", "SE_Bottom", "Bottom_CI95_lo", "Bottom_CI95_hi",
        "Top", "SE_Top", "Top_CI95_lo", "Top_CI95_hi",
        "R2", "AIC", "Fixed_Bottom", "Fallback_Pass2", "N_Removed",
        "Saturation_Corrected", "N_Wells_Corrected", "IC50_Delta_vs_Raw_uM",
        "Top_Anchor_Used"
    ]

    df_final = df_final.reindex(columns=cols)
    output_xlsx = os.path.join(folder_path, "Consolidated_IC50_Results.xlsx")
    df_final.to_excel(output_xlsx, index=False)
    
    if unregistered_drug_warnings:
        warning_path = os.path.join(folder_path, "UNREGISTERED_DRUGS_WARNING.txt")
        with open(warning_path, "w", encoding="utf-8") as f:
            f.write(f"{len(unregistered_drug_warnings)} block(s) were SKIPPED -- no fit "
                     "was attempted -- because their drug name did not match any row "
                     "in Starting_Concentrations.xlsx. Fitting these with a fabricated "
                     "concentration would produce a misleading IC50, so no plot or "
                     "result row was produced for them. Likely a spelling/naming "
                     "mismatch (e.g. 'Gemcitabina' in the filename vs 'Gemcitabine' in "
                     "the spreadsheet). Fix Starting_Concentrations.xlsx and re-run.\n\n")
            for w in unregistered_drug_warnings:
                f.write(f"- {w}\n")
        # Printed last, after the run's hundreds of other log lines, and written to
        # its own file on disk -- so it survives both an unattended/batch run where
        # nobody reads the console, and a later look at just the output folder.
        print("\n" + "!" * 70)
        print(f"  [UNREGISTERED DRUGS] {len(unregistered_drug_warnings)} block(s) were "
              f"SKIPPED -- no fit attempted -- because their drug name is not "
              f"registered in Starting_Concentrations.xlsx. "
              f"See {os.path.basename(warning_path)}.")
        for w in unregistered_drug_warnings:
            print(f"    - {w}")
        print("!" * 70)

    print("\n[COMPLETE SUCCESS] Scientific pipeline executed successfully.")
    print(f"  Results saved to: {output_xlsx}")
    print(f"  Total analyzed curves: {len(df_final)}")