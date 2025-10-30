# Malaysia Car Price Prediction [Kaggle](https://www.kaggle.com/code/insyirahazman/car-price-prediction-using-ml)

This repository contains a Jupyter notebook and small demo app that trains and evaluates models to predict used car prices in Malaysia. The primary notebook is `main.ipynb` and the dataset is `Malaysia_Final_CarList_Compiled.csv`.

## Objective
Predict the market price (`Price`) of used cars in Malaysia using vehicle attributes (Model, Year, Color, Engine.Cap, Mileage, etc.).

## Dataset
- File: `Malaysia_Final_CarList_Compiled.csv`
- The notebook drops non-essential columns (`Desc`, `Link`) and standardizes timestamps (`Updated`).

## Preprocessing highlights
- Converted `Updated` to datetime.
- Label-encoded categorical fields (`Model`, `Engine.Cap`, `Transm`, `Color`, `Car.Type`). Saved encoders so the deployed app can accept human-readable categories.
- Filled missing `Mileage` values with the median.
- Detected outliers via IQR and clipped extreme `Mileage` values.
- Removed duplicate rows.

## Exploratory analysis
- Visualized distributions and relationships between features and price (boxplots, scatter, heatmap).
- Found the strongest numeric correlation between `Price` and `Year` (≈ 0.81).

## Models evaluated
The notebook trains and compares these models using R2, MSE, MAE, and RMSE metrics:

- Simple Linear Regression (Year)
- Multiple Linear Regression (Model, Year, Color, Engine.Cap)
- Random Forest Regressor
- Decision Tree Regressor

## Best model
The Random Forest Regressor was the best performer in the notebook (reported R2 ≈ 87.15%). The notebook selects the best model by R2 and prints full metrics for comparison.

## Deployment (Gradio demo)
I included a lightweight Gradio demo so you can interactively check prediction accuracy.

Files added for deployment:
- `app_gradio.py` — single-prediction Gradio app (Model, Year, Color, Engine.Cap).
- `requirements.txt` — minimal dependencies for the demo (gradio, scikit-learn, pandas, numpy, joblib).

How it works (quick):
1. From the notebook, save the trained best model and the fitted LabelEncoders by running the "Save best model" cell. This produces two files in the project root:
   - `best_model.joblib`
   - `encoders.joblib`

2. Install dependencies and run the demo:

```powershell
pip install -r .\requirements.txt
python .\app_gradio.py
```

3. The Gradio UI accepts:
   - Model (human-readable name or numeric encoding)
   - Year (number)
   - Color (human-readable name or numeric encoding)
   - Engine.Cap (number)

Notes on encoders:
- `encoders.joblib` contains the LabelEncoder objects used when training. When present the demo will accept human-readable category names (e.g., color names or model names) and encode them automatically. If encoders are missing you must provide numeric encoded values for categorical features.

Testing & measuring accuracy in deployment
- Prepare a labeled CSV (holdout/test set) with the exact feature columns: `Model, Year, Color, Engine.Cap, Price`.
- You can test locally by running the demo and entering a single sample, or by writing a small script that loads `best_model.joblib` and predicts on the test CSV to compute metrics (MAE, RMSE, MAPE, R2).

Hosting options
- Local: run the Gradio app on your machine (good for testing and demos).
- Cloud options: deploy containerized app (FastAPI + Docker) or use Gradio/Streamlit hosting (Hugging Face Spaces for Gradio demos) for sharing.

## How to run the notebook
1. Open `main.ipynb` in Jupyter or VS Code.
2. Ensure `Malaysia_Final_CarList_Compiled.csv` is in the same folder.
3. Run all cells top-to-bottom. After training, run the save-cell to produce `best_model.joblib` and `encoders.joblib` for the demo.

## Files
- `main.ipynb` — main notebook containing EDA, preprocessing, modeling and evaluation.
- `app_gradio.py` — single-prediction Gradio demo.
- `requirements.txt` — Python packages required for the demo.
- `Malaysia_Final_CarList_Compiled.csv` — dataset.