# Airbnb Rental Market Price Analysis

An end-to-end data analysis and machine learning project focused on predicting Airbnb nightly rental prices and identifying key market drivers across NYC neighborhoods.

---

## Overview

This project analyzes Airbnb rental listing data to uncover patterns in pricing based on property capacity, location, room type, and availability. It features descriptive statistical analysis, data visualizations, feature engineering, and a tuned **Random Forest Regressor** model to predict nightly prices on a logarithmic scale.

---

## Key Findings & Insights

* **Primary Price Drivers:**
  * **Capacity & Room Type:** `accommodates`, `bedrooms`, and property structure (e.g., *Entire home/apt* vs. *Private room*) serve as the dominant numerical predictors of nightly price.
  * **Location:** Geographic regions (`neighbourhood_group`) heavily shift baseline pricing, with Manhattan listings commanding the highest price premium.
* **Model Calibration:**
  * Predictions for standard listings ($50–$250) are well-calibrated and show errors centered close to zero.
  * High-end luxury listings ($300+) show larger positive residuals, indicating high-end pricing relies on additional niche features (e.g., host badge status, specialized amenities).

---

## Dataset & Features

The dataset comprises **1,200 listings** with numerical, categorical, and geographical attributes:

| Feature Name | Description | Type |
| :--- | :--- | :--- |
| `price` | Target variable (nightly listing price in USD) | Numeric |
| `room_type` | Entire home/apt, Private room, Shared room, Hotel room | Categorical |
| `neighbourhood_group` | NYC borough (Manhattan, Brooklyn, Queens, Staten Island, Bronx) | Categorical |
| `accommodates` | Maximum number of guests allowed | Numeric |
| `bedrooms` | Number of bedrooms | Numeric |
| `bathrooms` | Number of bathrooms | Numeric |
| `minimum_nights` | Minimum stay requirement in nights | Numeric |
| `availability_365` | Available days per year | Numeric |
| `reviews_per_month` | Average monthly review count (missing values filled with `0`) | Numeric |

---

## Project Workflow

1. **Data Cleaning & Preprocessing**
   * Processed string formatting on the `price` column (removed `$` and `,` and converted to float).
   * Imputed missing values in `reviews_per_month` with `0`.
   * Applied log transformation (`np.log1p`) on `price` to correct right-skewness.

2. **Exploratory Data Analysis (EDA)**
   * Summary statistics grouped by `room_type`.
   * Outlier and price distribution visualizations using Seaborn boxplots.
   * Correlation heatmaps across numerical features.
   * Geographic mapping of listings below $500 using latitude and longitude coordinates.

3. **Machine Learning Modeling**
   * Encoded categorical variables (`room_type`, `neighbourhood_group`) using `pd.get_dummies()`.
   * Split dataset into 80% training and 20% testing sets.
   * Built a baseline **Random Forest Regressor** model.
   * Performed hyperparameter tuning using **GridSearchCV** across `n_estimators`, `max_depth`, and `min_samples_split`.

4. **Model Serialization**
   * Exported the best estimator model (`airbnb_price_model.pkl`) and corresponding feature column schema (`model_features.pkl`) using `joblib`.

---

## Model Performance

| Metric | Value |
| :--- | :--- |
| **Best Hyperparameters** | `{'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 50}` |
| **Tuned $R^2$ Score (log scale)** | **0.592** |
| **RMSE (actual dollar scale)** | **$85.44** |

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/pachareshreyash41-dev/airbnb-price-analysis.git](https://github.com/pachareshreyash41-dev/airbnb-price-analysis.git)
   cd airbnb-price-analysisM
