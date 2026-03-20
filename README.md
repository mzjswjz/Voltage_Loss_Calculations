# Voltage Loss Calculations

Analysis of voltage loss mechanisms in photovoltaic devices, including radiative, non-radiative, and Shockley-Queisser limit calculations.

## Overview

This repository contains calculations for understanding voltage losses in solar cells. The analysis covers:

- Radiative recombination limits
- Non-radiative recombination (Shockley-Read-Hall)
- Auger recombination
- Detailed balance analysis

## Theory

The open-circuit voltage (V_OC) of a solar cell is determined by the competition between:

1. **Radiative Recombination** - Fundamental limit from band-to-band recombination
2. **Non-Radiative Recombination** - Defect-assisted recombination (SRH)
3. **Auger Recombination** - Carrier-carrier interaction

The voltage loss is calculated as:

```
ΔV_OC = (kT/q) * ln(EQE_EL / EQE_SQ)
```

## Key Equations

### Radiative Limit
```
V_OC,rad = (kT/q) * ln(J_SC / J_0,rad + 1)
```

### Non-Radiative Loss
```
ΔV_OC,nr = (kT/q) * ln(EQE_EL / EQE_SQ)
```

### EQE_EL
```
EQE_EL = J_L / (q * φ_bb)
```

## Data Format

Input files should contain:
- Wavelength (nm)
- Incident photon flux
- Solar cell response

## Usage

```python
python voltage_loss.py --input data.csv --output results/
```

## Dependencies

- NumPy
- SciPy
- Matplotlib
- pandas

## Author

mzjswjz
