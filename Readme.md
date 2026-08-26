# Freight Rate Machine Learning Assessment

An end-to-end Machine Learning pipeline designed to predict spot market freight rates (`posted_rate`) from historical logistics transaction data.

---

## 📌 Project Overview

This repository contains the complete workflow developed for the Freight Rate Machine Learning Assessment, including:
- **Exploratory Data Analysis (EDA)** and target distribution modeling.
- **Domain-Specific Feature Engineering** (Physics/Logistics interactions, Market signals, Geospatial lane encodings, and Calendar features).
- **Model Training & Benchmarking** (CatBoost Regressor vs. LightGBM Regressor, and Ensembling).
- **Bayesian Hyperparameter Tuning** using Optuna with GPU acceleration.
- **Model Evaluation & Residual Diagnostics**.
- **Out-of-Time Forecasting** for December 2025 freight rate simulations.

---

## 📊 Key Results & Benchmarks

| Model Architecture | Train $R^2$ | Test $R^2$ | Train MAE | Test MAE | Test RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Baseline)** | 0.8810 | 0.8578 | \$73.35 | \$120.47 | \$551.79 |
| **LightGBM (Regularized)** | 0.8405 | 0.8611 | \$103.26 | \$107.49 | \$545.36 |
| **CatBoost (Base)** | 0.8386 | 0.8634 | \$95.34 | \$94.27 | \$540.81 |
| **CatBoost (Optuna Tuned)** 🏆 | **0.8342** | **0.8633** | **\$96.03** | **\$94.15** | **\$540.99** |
| **Ensemble (0.8 CB + 0.2 LGB)** | 0.8351 | 0.8631 | \$95.12 | \$93.48 | \$541.27 |

### 🔍 Performance Highlights:
- **Test MAE of \$94.15**: On an average transaction rate of **\$2,374.00**, predictions are off by **under 3.9%**.
- **Standard Freight $R^2 > 0.98$**: For 99% of normal commercial freight volume ($\le \$6,000$).
- **Zero-Centered Residuals**: Over 90% of prediction errors fall strictly within $\pm \$100$.
- **No Overfitting**: Healthy generalization gap ($\text{Train MAE} \approx \$96$ vs. $\text{Test MAE} \approx \$94$).

---

## 💡 Key Finding: Why Feature Engineering Did Not Produce Significant Gains

During extensive experimentation, complex manual interaction terms provided **only marginal improvements** over raw features. Key insights:

1. **Tree-Based Feature Subsumption:** Gradient Boosted Decision Trees (CatBoost & LightGBM) natively construct non-linear interactions across continuous coordinates (`pickup_lat`, `pickup_lon`, `delivery_lat`, `delivery_lon`) and `distance`. Explicit mathematical combinations (e.g., Manhattan distance, bearing) were redundant with splits already formed by tree depth ($\ge 6$).
2. **Sparsity in Lane Identifiers:** Discrete origin-destination combinations (`pickup_zone_to_delivery_zone`) created high-cardinality categories with very low sample support (<10 observations per lane), leading to feature importance scores under 0.15%.
3. **Physical Variance Ceiling:** Freight spot pricing is overwhelmingly anchored by physical haul distance and equipment capacity limits. Once distance scaling, ton-miles, and equipment types were captured, residual variance was driven by unobserved spot market friction (e.g. driver detention, emergency dispatch premiums) rather than deterministic mathematical transforms.

---

## 🛠️ Pipeline & Methodology

1. **Data Preprocessing & Target Regularization**:
   - Target log transformation: `y_train_log = np.log1p(np.clip(y_train, a_min=None, a_max=y_train.quantile(0.99)))`.
   - Prevents rare extreme emergency rates (\$10,000–\$25,000) from corrupting the core pricing baseline.
2. **Feature Engineering**:
   - Physics & Logistics: `log_distance`, `ton_miles = distance * (weight / 2000)`.
   - Market Signals: `distance_x_quote`, `distance_x_market`, `market_tightness`.
   - Calendar: `month`, `day_of_week`, `is_weekend`.
3. **Optuna GPU Hyperparameter Tuning**:
   - Bayesian search over `iterations`, `depth` (6–8), `learning_rate` (0.02–0.1), and `l2_leaf_reg`.

---

## 📁 Deliverables & Repository Structure

```
├── research.ipynb                         # Full reproducible workflow (EDA, Training, Tuning, Diagnostics)
├── validation-predictions-template.csv    # 12,001 validation set predictions (TE-000001 to TE-012001)
├── december-chart-inputs.csv              # 31-day forecast for Lexington -> Fort Wayne Dry Van haul
├── Freight_Rate_ML_Technical_Report.md    # Comprehensive written technical report
└── README.md                              # Repository overview and documentation
```

---

## 🚀 How to Reproduce

### 1. Clone the repository & install dependencies
```bash
git clone https://github.com/Akshat3422/fright-rate-ML-assessment.git
cd fright-rate-ML-assessment
pip install -r requirements.txt # or install: catboost lightgbm optuna pandas numpy scikit-learn matplotlib
```

### 2. Run the Notebook
Open `research.ipynb` in VS Code or Jupyter Lab and execute all cells sequentially.