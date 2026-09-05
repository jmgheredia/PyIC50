import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats as scipy_stats
import warnings
from scipy.optimize import OptimizeWarning

# ─────────────────────────────────────────────────────────────────────────────
#  1. GLOBAL CONFIGURATION & METADATA
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    """Centralized, scientifically justified configuration parameters."""
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
def parse_filename(filename, conc_dict):
    """Parses plate design configuration from the filename."""
    base_name = os.path.basename(filename)
    name_no_ext = os.path.splitext(base_name)[0]

    if ";" not in name_no_ext:
        return None, None

    lines_part, drugs_part = name_no_ext.split(";", 1)
    cell_lines = [l.strip() for l in lines_part.split(",") if l.strip()]
    while len(cell_lines) < 1:
        cell_lines.append("CellLine_1")
    cell_lines = cell_lines[:2]

    drugs = _extract_drugs_from_string(drugs_part, conc_dict)
    while len(drugs) < 1:
        drugs.append({
            'name': 'UnknownDrug_1',
            'volume': 2,
            'base_conc': 100.0,
            'exact_conc': False,
            'search_index': 999
        })

    return cell_lines, drugs[:2]

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

def build_assignments(cell_lines, drugs, df_clean):
    """Splits raw plate layout into comparative experimental blocks."""
    n_lines = len(cell_lines)
    n_drugs = len(drugs)

    line1 = cell_lines[0]
    line2 = cell_lines[1] if n_lines >= 2 else cell_lines[0]
    drug1 = drugs[0]
    drug2 = drugs[1] if n_drugs >= 2 else drugs[0]

    return [
        (line1, drug1, df_clean.iloc[0:4]),
        (line2, drug2, df_clean.iloc[4:8]),
    ]

def create_anchor_points(df_valid, init_conc, anchor_value):
    """Generates a pseudo 'Well -1' at high concentration to anchor plateaus."""
    fict_rows = []
    fict_conc = init_conc * 3.0
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
            "is_fictional": True
        })
    return pd.DataFrame(fict_rows)


# ─────────────────────────────────────────────────────────────────────────────
#  4. WEIGHTED MODEL FITTING & INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def run_single_fit(X, Y, init_conc, control_mean, pseudo_zero_conc, fixed_bottom=None):
    """Fits the 4PL curve using Weighted Least Squares (WLS)."""
    unique_x = np.unique(X)
    weights = np.ones_like(Y, dtype=float)
    
    global_sd = np.std(Y) if len(Y) > 0 else 1.0
    floor_sd = 0.05 * global_sd if global_sd > 0 else 0.01

    for ux in unique_x:
        mask = (X == ux)
        reps = Y[mask]
        if len(reps) > 1:
            sd = np.std(reps)
            weights[mask] = 1.0 / (max(sd, floor_sd)**2)
        else:
            weights[mask] = 1.0 / (floor_sd**2)

    sigma = 1.0 / np.sqrt(weights)

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
                                                 absolute_sigma=True, maxfev=15000)
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
                                       absolute_sigma=True, maxfev=15000)
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
def _prepare_plate_data(df_drug, base_conc, volume, exact_conc=False):
    """Initial QC, MAD outlier detection, and pipetting validation."""
    if exact_conc:
        init_conc = base_conc
    else:
        init_conc = base_conc * (volume / 2.0)
    dilution_factor = 3.0

    concs_raw = []
    c = init_conc
    for i in range(11):
        concs_raw.append(c)
        c /= dilution_factor
    pseudo_zero_conc = concs_raw[-1] / 100.0

    pts = []
    for r_idx, row in df_drug.iterrows():
        for col_idx in range(12):
            conc = concs_raw[col_idx] if col_idx < 11 else pseudo_zero_conc
            pt_type = "dilution" if col_idx < 11 else "control"
            pts.append({
                "orig_idx": len(pts),
                "row_idx": r_idx,
                "col_idx": col_idx,
                "concentration": conc,
                "value": row.iloc[col_idx],
                "type": pt_type,
                "is_mad_outlier": False
            })
    df_pts = pd.DataFrame(pts)

    for col_idx in range(12):
        col_mask = df_pts["col_idx"] == col_idx
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
                  fixed_bottom, use_fixed_bottom):
    """
    Main model optimization pipeline:
    1. Perform initial robust weighted fit.
    2. Check for individual experimental outliers via 95% Prediction Band 
       to clean bimodal replicate failures (e.g., pipetting drops) before LOO.
    3. Run iterative LOO loop on the cleaned dataset optimized by AIC.
    """
    results = []
    df_work = df_valid.copy()
    k_params = 3 if fixed_bottom is not None else 4

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
        **stats0,
        "IC50_Boot_lo": np.nan,
        "IC50_Boot_hi": np.nan,
        "IC50_Boot_median": np.nan,
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

    if len(results) > 0:
        b_lo, b_hi, b_med = _bootstrap_ic50(df_work, init_conc, control_mean,
                                             pseudo_zero_conc, fixed_bottom)
        results[-1]["IC50_Boot_lo"]     = b_lo
        results[-1]["IC50_Boot_hi"]     = b_hi
        results[-1]["IC50_Boot_median"] = b_med

    return results

# ─────────────────────────────────────────────────────────────────────────────
#  7. EXPERIMENTAL PIPELINE PASSES
# ─────────────────────────────────────────────────────────────────────────────
def analyze_drug_pass1(df_drug, drug_name, volume, base_conc, file_label,
                       exact_conc=False):
    """PASSED 1: Adaptive algorithm utilizing localized data constraints."""
    df_pts, df_valid, control_mean, init_conc, pseudo_zero_conc = \
        _prepare_plate_data(df_drug, base_conc, volume, exact_conc=exact_conc)

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
                
                df_fict = create_anchor_points(df_valid, init_conc, local_bottom)
                df_valid_fit = pd.concat([df_valid, df_fict], ignore_index=True)

                results = _run_loo_fits(df_valid_fit, init_conc, control_mean,
                                        pseudo_zero_conc,
                                        fixed_bottom=local_bottom,
                                        use_fixed_bottom=True)
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
                                            use_fixed_bottom=False)
        else:
            if r2 >= Config.R2_POOR_FIT:
                reason = f"R²={r2:.3f} OK"
            else:
                reason = "No cytotoxic response curve recognized."
            print(f"    [Pass 1] {drug_name}: {reason} -> fit processed.")

            if should_anchor and r2 < Config.R2_POOR_FIT:
                print(f"    [Well -1] {drug_name}: {n_dead_anchor} replicates "
                      f"clearly dead at high concentrations -> anchoring.")
                df_fict = create_anchor_points(df_valid, init_conc, anchor_value)
                df_valid_anchor = pd.concat([df_valid, df_fict], ignore_index=True)
                results = _run_loo_fits(df_valid_anchor, init_conc, control_mean,
                                        pseudo_zero_conc,
                                        fixed_bottom=anchor_value,
                                        use_fixed_bottom=True)
                for r in results:
                    r["Fallback_Pass2"] = False
                meta["needs_fallback"] = False
                if popt[1] > 0:
                    meta["bottom_top_ratio"] = anchor_value / popt[1]
                meta["df_valid"] = df_valid_anchor
            else:
                results = _run_loo_fits(df_valid, init_conc, control_mean,
                                        pseudo_zero_conc, fixed_bottom=None,
                                        use_fixed_bottom=False)

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

    print(f"    [Pass 2] {drug_name}: fixed Bottom reference = {ref_bottom:.4f}")

    results = []
    if meta.get("late_abrupt_drop", False):
        df_fict = create_anchor_points(df_valid, init_conc, ref_bottom)
        df_valid = pd.concat([df_valid, df_fict], ignore_index=True)
        meta["df_valid"] = df_valid

    try:
        results = _run_loo_fits(df_valid, init_conc, control_mean,
                                pseudo_zero_conc, fixed_bottom=ref_bottom,
                                use_fixed_bottom=True)
        for r in results:
            r["Fallback_Pass2"] = True
    except (RuntimeError, ValueError) as e:
        print(f"    [Pass 2] Execution error on {drug_name} in {file_label}: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  8. GRAPHICAL EXPORT PANEL
# ─────────────────────────────────────────────────────────────────────────────
def save_single_plot(res, valid_df, drug_name, volume, cell_line, file_name, folder_path):
    """Generates standardized, high-resolution scientific plots."""
    if not res:
        return

    run0         = res[0]
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

    # Points purged by the iterative vertical prediction-band filter
    # (pre-LOO) should not be shown as ordinary valid points — they were
    # excluded from the fit used to draw both curves below.
    vp_points = run0.get("Vertical_Purge_Points", [])
    if vp_points:
        vp_set = {(round(p["concentration"], 10), round(p["value"], 10)) for p in vp_points}
        real_df_key = list(zip(real_df["concentration"].round(10), real_df["value"].round(10)))
        keep_mask = [k not in vp_set for k in real_df_key]
        real_df = real_df[keep_mask]

    ax.set_xscale('log')
    ax.set_xlim(Config.X_LIM_MIN, Config.X_LIM_MAX)

    x_c = np.linspace(np.log10(Config.X_LIM_MIN), np.log10(Config.X_LIM_MAX), 500)

    ax.scatter(real_df["concentration"], real_df["value"],
               color='#2ec4b6', alpha=0.55, s=32, zorder=2, label='Valid points')

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
    print("  The script assumes a continuous 3-fold dilution starting from Column 1")
    print("  (Well 1) to Column 11. Do NOT leave Well 1 empty or start the series")
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

    # Detect layout configuration
    _has_volume_col = len(df_conc.columns) >= 3
    if _has_volume_col:
        print("  [Starting Concentrations] Standard 3-column format verified: (Drug, Volume, Concentration)")
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
        else:
            val = float(row.iloc[1])
            conc_dict[name] = val

    xls_files = glob.glob(os.path.join(folder_path, "*.xls"))
    xls_files = [f for f in xls_files if not os.path.basename(f).startswith("~$")]
    print(f"Found {len(xls_files)} plate files to analyze.")

    skipped_files = []
    pass1_results = []
    ref_bottoms = {}

    # ── PASS 1 ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PASS 1: Adaptive, free fit for all datasets")
    print("="*60)

    for file in xls_files:
        file_name = os.path.basename(file)
        print(f"\nProcessing: {file_name}")

        cell_lines, drugs = parse_filename(file_name, conc_dict)
        if cell_lines is None:
            print(f"  [SKIPPED] Missing ';' in filename. Format must be '<cell_lines>; <drugs>.xls'")
            skipped_files.append(file_name)
            continue

        df_plate = pd.read_excel(file)
        df_clean = df_plate.dropna(how='all')
        if len(df_clean) < 8:
            print(f"  [WARNING] Insufficient data rows. skipping.")
            skipped_files.append(file_name)
            continue

        print(f"  Lines: {', '.join(cell_lines)}")
        for i, d in enumerate(drugs, 1):
            print(f"  Drug {i}: {d['name']} ({d['volume']} uL, base {d['base_conc']} uM)")

        assignments = build_assignments(cell_lines, drugs, df_clean)

        for cell_line, drug, df_rows in assignments:
            print(f"  → {cell_line} / {drug['name']}")
            res, df_pts, df_valid, meta = analyze_drug_pass1(
                df_rows, drug['name'], drug['volume'], drug['base_conc'], file_name,
                exact_conc=drug.get('exact_conc', False)
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

            # Accumulate ratio using the Median for robust fallback statistics
            if res and meta["r2_free"] is not None:
                if (meta["r2_free"] >= Config.R2_POOR_FIT
                        and meta["bottom_top_ratio"] is not None):
                    ref_bottoms.setdefault(cell_line, []).append(
                        meta["bottom_top_ratio"]
                    )
                    print(f"    → Bottom/Top Ratio={meta['bottom_top_ratio']:.4f} added for '{cell_line}'")

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
                    meta["pseudo_zero_conc"], fixed_bottom=None, use_fixed_bottom=False
                )
                for r in res_fallback:
                    r["Fallback_Pass2"] = False
                    r["Fixed_Bottom"] = False
                entry["results"] = res_fallback
            except (RuntimeError, ValueError) as e:
                print(f"    Error in fallback free fit: {e}")

    # ── CONSOLIDATE RESULTS ──────────────────────────────────────────────────
    all_runs_data = []
    for entry in pass1_results:
        for r in entry["results"]:
            r.update({
                "File": entry["file_name"],
                "Cell_Line": entry["cell_line"],
                "Drug": entry["drug"]["name"],
                "Stock_Volume_uL": entry["drug"]["volume"],
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
        "Well1_Concentration_uM", "Run", "Removed_Point",
        "IC50_uM", "IC50_CI95_lo", "IC50_CI95_hi",
        "IC50_Boot_lo", "IC50_Boot_hi", "IC50_Boot_median",
        "SE_IC50_uM",
        "HillSlope", "SE_HillSlope", "Hill_CI95_lo", "Hill_CI95_hi",
        "Bottom", "SE_Bottom", "Bottom_CI95_lo", "Bottom_CI95_hi",
        "Top", "SE_Top", "Top_CI95_lo", "Top_CI95_hi",
        "R2", "AIC", "Fixed_Bottom", "Fallback_Pass2", "N_Removed"
    ]

    df_final = df_final.reindex(columns=cols)
    output_xlsx = os.path.join(folder_path, "Consolidated_IC50_Results.xlsx")
    df_final.to_excel(output_xlsx, index=False)
    
    print("\n[COMPLETE SUCCESS] Scientific pipeline executed successfully.")
    print(f"  Results saved to: {output_xlsx}")
    print(f"  Total analyzed curves: {len(df_final)}")