# Prognos — Clinical Disease Prediction System

A production-grade machine learning pipeline for early-stage disease risk assessment across three clinical domains: breast cancer, cardiovascular disease, and Type 2 diabetes. Built on UCI ML Repository datasets using a soft-voting ensemble of four algorithms with engineered features, class-imbalance correction, and a standalone web interface.

---

## Overview

Prognos implements a full end-to-end ML workflow — from raw dataset ingestion to a deployable prediction interface — following practices standard in clinical decision-support research. The system achieves **99.1% accuracy on breast cancer classification**, **AUC-ROC of 0.999**, and competitive results on the notoriously noisy Pima Diabetes dataset.

The web frontend runs entirely client-side with no backend dependency, and a Flask API integration path is available for connecting directly to the trained model artifacts.

---

## Results

| Dataset | Accuracy | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Breast Cancer (Wisconsin) | **99.1%** | **0.999** | 0.99 | 0.99 | 0.99 |
| Heart Disease (Cleveland) | **86.9%** | **0.959** | 0.89 | 0.87 | 0.87 |
| Diabetes (Pima Indian) | **77.3%** | **0.833** | 0.79 | 0.77 | 0.78 |

Diabetes accuracy at 77.3% is consistent with published benchmarks on this dataset (76–80%), bounded by sample size (768 rows) and systematic noise from zero-value placeholders for missing biological measurements.

---

## Datasets

| Dataset | Source | Samples | Features | Task |
|---|---|---|---|---|
| Wisconsin Breast Cancer | UCI ML Repository | 569 | 30 | Binary (Malignant / Benign) |
| Cleveland Heart Disease | UCI ML Repository | 303 | 13 | Binary |
| Pima Indian Diabetes | UCI / Kaggle | 768 | 8 | Binary |

---

## ML Pipeline

### Preprocessing

Missing value imputation uses a median strategy. The Heart Disease dataset encodes missing values as `?`; the Diabetes dataset encodes physiologically impossible zero values across glucose, BMI, insulin, blood pressure, and skin thickness — all replaced with per-column medians prior to scaling.

Class imbalance is addressed using SMOTETomek — a combined oversampling and Tomek link cleaning approach applied exclusively to the training split to prevent synthetic samples from contaminating held-out evaluation.

Features are scaled using StandardScaler fitted on training data only, with the fitted scaler persisted alongside the model artifact for inference-time consistency. All splits are stratified 80/20 to preserve class distribution across train and test sets.

### Feature Engineering

**Breast Cancer** — mean-to-worst ratios for all 10 nucleus measurement pairs (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), effectively doubling the informative feature count.

**Heart Disease** — cardiovascular interaction terms including age-by-thalach, cholesterol-to-age ratio, and oldpeak-slope product. Binary risk flag for age above 55 and a CA-thal interaction term.

**Diabetes** — insulin resistance proxy derived from insulin, BMI, and glucose. Binary risk flags for high glucose and high BMI. Polynomial glucose term and interaction terms across pregnancies, age, glucose, and BMI.

### Ensemble Architecture

Each disease model is a soft-voting ensemble aggregating predicted class probabilities across five base estimators before taking argmax. Soft voting consistently outperforms hard voting on imbalanced clinical data by producing better-calibrated outputs.

| Algorithm | Role |
|---|---|
| XGBoost | Primary boosting estimator — highest individual accuracy |
| Gradient Boosting | Secondary boosting — diversity through different base learners |
| Random Forest | Bagging estimator — robust to outliers, strong feature importance |
| Support Vector Machine | Margin-based classifier — strong on scaled high-dimensional data |
| Logistic Regression | Linear baseline — regularisation anchor, improves ensemble calibration |

Cross-validation uses 5-fold stratified KFold. Hyperparameter search uses RandomizedSearchCV with 20 iterations — sufficient to match GridSearchCV performance at a fraction of the compute cost.

---

## Project Structure

```
prognos/
├── notebooks/
│   └── disease_prediction.ipynb    # Full training and evaluation pipeline
├── models/
│   ├── breast_cancer_v2.pkl        # Trained ensemble — breast cancer
│   ├── heart_disease_v2.pkl        # Trained ensemble — heart disease
│   ├── diabetes_v2.pkl             # Trained ensemble — diabetes
│   ├── bc_scaler.pkl               # Fitted StandardScaler — breast cancer
│   ├── hd_scaler.pkl               # Fitted StandardScaler — heart disease
│   └── db_scaler.pkl               # Fitted StandardScaler — diabetes
├── prognos.html                    # Standalone web interface
└── README.md
```

---

## Web Interface

`prognos.html` is a self-contained single-file application requiring no server or build step. It runs directly in any modern browser and includes full input forms for all three disease models, animated probability output, risk stratification (High / Moderate / Low), per-prediction doughnut risk chart, radar and bar performance visualisations, and an animated intro loader.

The interface can be connected to trained model artifacts by pointing the client-side prediction calls to a Flask API serving the `.pkl` files — enabling real model inference rather than the client-side logistic approximation used in standalone mode.

---

## Limitations

**Breast Cancer** — the 30-feature FNA dataset is well-structured and high-signal. 99.1% accuracy is reproducible but may not generalise to other imaging modalities, scanner types, or demographic groups outside the Wisconsin cohort.

**Heart Disease** — the Cleveland dataset contains only 303 samples. The model is not validated on the other UCI heart disease cohorts (Hungarian, Switzerland, VA Long Beach) and cross-cohort generalisation is not guaranteed.

**Diabetes** — the Pima dataset is restricted to female patients of Pima Indian heritage aged 21 and above. Predictions for other demographic groups are extrapolations beyond the training distribution. The 77–78% accuracy ceiling is well-established in the literature and cannot be significantly exceeded without external data augmentation or a larger, cleaner dataset.

All models operate on structured tabular clinical data only. They do not process medical imaging, ECG time series, or unstructured clinical notes.

---

## Dependencies

Python 3.11+, scikit-learn, XGBoost, imbalanced-learn, pandas, numpy, matplotlib, seaborn, joblib, jupyter. macOS users require `libomp` (via Homebrew) for XGBoost OpenMP support.

---

## References

- Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine.
- Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
- Chawla, N.V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357.
- Tomek, I. (1976). Two Modifications of CNN. *IEEE Transactions on Systems, Man, and Cybernetics*, 6, 769–772.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

---

## License

MIT License. This project is intended for research and educational purposes only. It is not a certified medical device and must not be used as a substitute for professional clinical diagnosis.
