# High-Accuracy Prediction of Vessels’ Estimated Time of Arrival in Seaports: A Hybrid Machine Learning Approach

[![DOI:10.1016/j.martra.2025.100133](https://img.shields.io/badge/DOI-10.1016%2Fj.martra.2025.100133-blue)](https://www.sciencedirect.com/science/article/pii/S2666822X2500005X)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

This repository contains the official implementation of the paper **“High-accuracy prediction of vessels' estimated time of arrival in seaports: A hybrid machine learning approach”**, published in *Maritime Transport Research* (2025).  
We introduce a novel tree-based stacking ensemble that integrates **Extra Trees**, **AutoGluon Tabular**, and **LightGBM** as base learners, with a **Random Forest Regressor** meta-learner, achieving **MAPE = 0.25 %** on Baltic Sea AIS data.

---

## Overview

Optimising the Estimated Time of Arrival (ETA) of vessels is critical for port efficiency, congestion reduction, and sustainable maritime logistics. This work develops a hybrid regression stacking model that exploits the strengths of multiple tree-based algorithms to deliver state-of-the-art ETA predictions on real-world Automatic Identification System (AIS) data from the Baltic Sea.

Key highlights:
- **MAPE of 0.25 %** and **R² of 0.999686** – outperforming all individual models and published benchmarks.
- **Robustness** validated via 5-fold cross-validation, ablation studies, and feature importance analysis.
- **Fully reproducible pipeline** from raw AIS data ingestion to final ensemble evaluation.

---

## Citation

If you use this code or build upon our work, please cite:

```bibtex
@article{saber2025eta,
  title     = {High-accuracy prediction of vessels' estimated time of arrival in seaports: A hybrid machine learning approach},
  author    = {Saber, Sunny Md. and Thowai, Kya Zaw and Rahman, Muhammad Asifur and Hassan, Md. Mehedi and Bari, A.B.M. Mainul and Raihan, Asif},
  journal   = {Maritime Transport Research},
  volume    = {8},
  pages     = {100133},
  year      = {2025},
  doi       = {10.1016/j.martra.2025.100133}
}
```

- **Paper link:** [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666822X2500005X)

---

## Repository Structure

```
seaport-eta-hybrid-ml/
├── data/                       # Raw & processed AIS data (not included)
├── notebooks/                  # Exploratory data analysis
├── src/
│   ├── preprocessing.py        # Data cleaning, trajectory construction, geofencing
│   ├── feature_engineering.py  # Distance calculation (Ramer–Douglas–Peucker, Haversine)
│   ├── models.py               # Individual regressors with hyperparameter tuning
│   ├── stacking_ensemble.py    # Stacking ensemble training
│   ├── evaluation.py           # Metrics, cross-validation, SHAP analysis
│   └── utils.py                # Helper functions and plotting
├── results/                    # Saved models, figures, metrics tables
├── requirements.txt            # Python dependencies
├── LICENSE
└── README.md
```

---

## Methodology Summary

1. **Data & Preprocessing**  
   - Public **Baltic Sea AIS dataset** (Hakola, 2020, IEEE Dataport) – 14 M records, 49 399 voyages.  
   - Cleaning: remove stationary, duplicate, ultra‑short, and erroneous trajectories; geofence ports; standardise time formats.  
   - Final dataset: **18 769 voyages** ≥ 300 nautical miles.

2. **Feature Engineering**  
   - **Dynamic features:** average speed (SOG), speed standard deviation, course, voyage distance (Ramer‑Douglas‑Peucker + Haversine).  
   - **Vessel characteristics:** type, breadth, draft.  
   - **Spatiotemporal:** departure hour, coordinates.  
   - (Weather variables excluded – speed and draft already reflect environmental influences.)

3. **Models & Hyperparameter Tuning**  
   - 10 regressors: Linear Regression, KNN, Random Forest, XGBoost, LightGBM, CatBoost, NGBoost, Extra Trees, AutoGluon Tabular.  
   - **Bayesian Optimization** for all non‑AutoGluon models; AutoGluon uses internal AutoML.

4. **Stacking Ensemble (Best Model)**  
   - **Base learners:** Extra Trees, LightGBM, AutoGluon Tabular  
   - **Meta‑learner:** Random Forest Regressor  
   - Trained with 5‑fold out‑of‑fold predictions; ≈ 214 M parameters.

5. **Evaluation**  
   - **Metrics:** MAPE, MAE, MSE, R²  
   - **Validation:** 5‑fold CV, test‑set (20 %), ablation studies (removing base learners / changing meta‑learner), SHAP analysis.

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Packages listed in `requirements.txt`

### Installation

```bash
git clone https://github.com/kyazaw/seaport-eta-hybrid-ml.git
cd seaport-eta-hybrid-ml
pip install -r requirements.txt
```

### Data Download

Download the AIS dataset from IEEE Dataport:  
**DOI:** [10.21227/j3b5-es69](https://dx.doi.org/10.21227/j3b5-es69)

Place the raw CSV files in `data/raw/`. The preprocessing scripts expect the original format.

### Run the Pipeline

Execute the scripts in order:

```bash
python src/preprocessing.py        # Data cleaning & trajectory building
python src/feature_engineering.py  # Feature creation (distance, speed, etc.)
python src/models.py               # Individual model training & tuning
python src/stacking_ensemble.py    # Stacking ensemble training
python src/evaluation.py           # Final metrics, cross‑validation, SHAP
```

Outputs (models, plots, metrics tables) are saved in `results/`.

---

## Key Results

### Test‑Set Performance (Best Ensemble)

| Model                                      | MAPE (%) | MAE (s) | R²      |
|--------------------------------------------|----------|---------|---------|
| **Extra Trees + AutoGluon + LightGBM → RFR** | **0.25** | 1 500   | 0.999686 |

### Feature Importance (SHAP)

Top predictors: **average speed > voyage distance > average course** – together explain > 60 % of predictive power. Vessel type, breadth, and departure/arrival coordinates provide additional refinement.

### Robustness

| Ablation / Variation                              | MAPE (%) | MAE (s) |
|---------------------------------------------------|----------|---------|
| Remove Extra Trees (LightGBM + AutoGluon)         | 0.32     | 1 905   |
| Remove LightGBM (Extra Trees + AutoGluon)         | 0.32     | 1 824   |
| Remove AutoGluon (Extra Trees + LightGBM)         | 0.32     | 1 956   |
| Meta‑learner: Linear Regression                   | 0.99     | 3 699   |
| Meta‑learner: XGBoost                             | 0.50     | 2 757   |
| **Proposed (RFR meta‑learner)**                   | **0.25** | 1 500   |

Detailed tables, learning curves, scatter plots, and cross‑validation results can be found in the `results/` folder.

---

## License

This project is released under the MIT License. See `LICENSE` for details.

---

*Advancing smart port management through accurate, interpretable, and robust vessel ETA predictions.*
