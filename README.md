# Malaysia Car Price Prediction

This project builds a machine learning pipeline to predict used car market prices in Malaysia using a compiled dataset and exploratory data analysis. The main work is in the Jupyter notebook `car-price-prediction-using-ml.ipynb`.

## Objective
Predict the market price (`Price`) of used cars in Malaysia from vehicle attributes and basic market features.

## Dataset
- File: `Malaysia_Final_CarList_Compiled.csv`
- The notebook loads the CSV, drops non-essential columns (`Desc`, `Link`), and standardizes the `Updated` timestamp.

## Key data processing steps
- Converted `Updated` to datetime.
- Encoded categorical fields (`Model`, `Engine.Cap`, `Transm`, `Color`, `Car.Type`) using LabelEncoder.
- Filled missing values in `Mileage` (median imputation).
- Detected outliers via IQR and clipped extreme `Mileage` values to bounds.
- Removed duplicate rows.

## Exploratory analysis highlights
- Visualized price distributions and relationships between features and `Price`.
- Noted the strongest numeric correlation: `Price` with `Year` (≈ 0.81).
- Target distribution checked for skewness and visualized (histograms, KDE, scatter plots).

## Models evaluated
The notebook trains and evaluates the following models (using R2, MSE, MAE, RMSE):

- Simple Linear Regression (single feature: `Year`)
- Multiple Linear Regression (features: `Year`, `Color`, `Engine.Cap`)
- Random Forest Regressor
- Decision Tree Regressor

## Best model
According to the notebook's evaluation, the Random Forest Regressor performed best (reported R2 ≈ 87.15%). The notebook prints model metrics for all trained models and selects the best by R2 score.

## Recommendations & next steps
- Perform additional feature engineering (e.g., extract age from `Year`, encode brand/model differently, derive interaction terms).
- Consider more robust handling of outliers (per-feature strategies) and try scaling/normalization where appropriate.
- Hyperparameter tuning (GridSearchCV or RandomizedSearchCV) for tree-based models.
- Add cross-validation and more comprehensive train/validation splits to avoid overfitting.

## How to run
1. Open the notebook `main.ipynb` in Jupyter or VS Code.
2. Ensure `Malaysia_Final_CarList_Compiled.csv` is in the same project folder.
3. Run the cells top-to-bottom; required Python packages include (but are not limited to): pandas, numpy, matplotlib, seaborn, scikit-learn, scipy.

## Files
- `main.ipynb` — main notebook containing EDA, preprocessing, modeling and evaluation.
- `Malaysia_Final_CarList_Compiled.csv` — dataset used by the notebook.