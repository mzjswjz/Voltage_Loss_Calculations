# Voltage Loss Calculations

Python module for calculating voltage losses in organic/perovskite solar cells using EQE data and Shockley-Queisser theory.

## Files

| File | Description |
|------|-------------|
| `functions.py` | Core calculation functions: SQ limits, Jsc, J0 (radiative and non-radiative), Voc calculations |
| `utils.py` | Utility functions: black-body radiation, AM1.5G spectrum loading, linear fitting |
| `utils_EQE.py` | EQE data processing utilities: file loading, interpolation, smoothing, plotting |
| `Step1_EQE raw data trial.ipynb` | Notebook for initial EQE data exploration and visualization |
| `Step2_EQE raw data fit.ipynb` | Notebook for fitting EQE data and extracting parameters |
| `Step3_Calculate DB Voltage loss.ipynb` | Notebook for calculating detailed balance voltage losses |
| `ASTMG173.csv` | ASTM G173 standard solar reference spectrum data |
| `bins.csv` | Configuration file for data binning |

## Main Functions

### In `functions.py`

| Function | Purpose |
|----------|---------|
| `SQ(Eg)` | Calculate Shockley-Queisser limit (Voc, Jsc, FF, PCE) for a given band gap |
| `calculate_Jsc(E, EQE)` | Calculate short-circuit current density from EQE data |
| `J0_rad(EQE_df, phi_bb_df)` | Calculate radiative limit of saturation current density |
| `J0(EQE_df, phi_bb_df, EQE_EL)` | Calculate saturation current density including LED quantum efficiency |
| `J0_CT(EQE_EL, ECT, l, f)` | Calculate J0 from CT state properties (Vandewal method) |
| `Voc(Jsc, J0)` | Calculate open-circuit voltage |
| `Voc_rad(Voc, Jsc, J0_rad)` | Calculate radiative Voc limit and non-radiative voltage losses |

### In `utils.py`

| Function | Purpose |
|----------|---------|
| `bb(energy)` | Calculate black-body radiation spectrum |
| `getAM15()` | Load AM1.5G solar spectrum data |
| `linear(x, a, b)` | Linear function for fitting |

### In `utils_EQE.py`

| Function | Purpose |
|----------|---------|
| `load_EQE(file_id)` | Load and interpolate EQE data from file |
| `plot_EQE(EQE_df)` | Plot EQE spectra |
| `smooth_EQE(EQE_df, window)` | Apply Savitzky-Golay smoothing |

## Usage

Import functions in your scripts or run the Jupyter notebooks sequentially:
1. Step 1: Explore raw EQE data
2. Step 2: Fit and process EQE data
3. Step 3: Calculate voltage losses

---
*Author: mzjswjz*
