# PyIC50: Robust & Reproducible IC50 Estimation Pipeline

`PyIC50` is a high-performance, robust, and reproducible Python pipeline designed to automate the estimation of $IC_{50}$ values from 96-well cell viability assays. 

By prioritizing statistical rigor over arbitrary curve-cleaning, `PyIC50` provides a scientifically sound alternative to manual outlier dropping and standard unweighted fitting tools.

## 🔬 Core Scientific Features

* **Weighted Least Squares (WLS) Fitting:** Accounts for experimental heteroscedasticity by applying fitting weights inversely proportional to replicate variance at each dose level.
* **AIC-Optimized Outlier Removal:** Replaces blind $R^2$ optimization in the *Leave-One-Out* (LOO) loop with the **Akaike Information Criterion (AIC)**. This mathematically penalizes data loss, preventing overfitting and preserving biological noise.
* **95% Prediction Band Filtering:** Dynamically detects systematic pipetting failures or bimodal replicate drops (e.g., 50/50 splits) by projecting vertical prediction bands based on the $4PL$ model's Jacobian matrix.
* **Honest Bootstrap Uncertainty Estimation:** Computes parametric and non-parametric bootstrap confidence intervals (95% CI) over 1,000 iterations to accurately reflect experimental and pipeline-induced uncertainty.
* **Contextual Plateau Anchoring (Two-Pass Fallback):** Automatically identifies "late drop" curves and safely anchors the lower plateau using local biological constraints or historical median cell-line benchmarks.

## 🛠️ Prerequisites & Installation

`PyIC50` is designed to run as a standalone, zero-setup script. You only need to clone this repository and install the standard scientific Python stack:

```bash
pip install numpy pandas matplotlib scipy openpyxl