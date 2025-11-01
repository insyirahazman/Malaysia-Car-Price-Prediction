# 🚗 Malaysia Car Price Prediction (Perodua Brand)

**Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/insyirazman/car-price-prediction)  
**Notebook on Kaggle:** [View Notebook](https://www.kaggle.com/code/insyirahazman/perodua-car-price-prediction)  

https://github.com/user-attachments/assets/88e46061-631d-4a2b-a9d0-17d012d68501

---

## 🧭 Project Overview

This project aims to **predict the resale price of Perodua cars in Malaysia** using real-world marketplace data.  
It demonstrates the complete data science workflow from **data cleaning and EDA** to **model training and deployment** with **Gradio** and **Hugging Face Spaces**.

---

## 🎯 Objective

Estimate the **resale market price** (`Price`) of used Perodua cars based on:

- Model  
- Year  
- Color  
- Engine Capacity  
- Mileage  
- Transmission  
- Car Type  

---

## 🧩 Dataset

- **File:** `Malaysia_Final_CarList_Compiled.csv`  
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/norazrinnatasha/malaysia-car-list-price)  
- **Size:** ~10K+ rows of used car listings  
- **Target Variable:** `Price`

### Data Cleaning Steps
- Dropped irrelevant columns (`Desc`, `Link`)
- Converted `Updated` column to datetime
- Label-encoded categorical features (`Model`, `Engine.Cap`, `Transm`, `Color`, `Car.Type`)
- Filled missing `Mileage` values with median
- Detected and clipped outliers via IQR
- Removed duplicate rows

> 💡 **Suggestion:** Add a small table or image showing missing value counts before and after cleaning.

---

## 📊 Exploratory Data Analysis (EDA)

Key insights from the data:
- `Price` has a strong positive correlation with `Year` (~0.81)
- Newer models and higher engine capacities tend to have higher prices
- Mileage negatively correlates with price
- White and silver cars are more common across listings

### 🔍 Recommended Visualizations
You can include these plots for better insight:
- Boxplot: Price vs. Model  
- Scatterplot: Year vs. Price  
- Histogram: Price Distribution  
- Heatmap: Feature Correlation

```python
# Example: Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

---

## Models evaluated
The notebook trains and compares these models using R2, MSE, MAE, and RMSE metrics:

- Simple Linear Regression (Year)
- Multiple Linear Regression (Model, Year, Color, Engine.Cap)
- Random Forest Regressor
- Decision Tree Regressor

## Best model
The notebook selects the best model by R2 and saves it to `best_model.joblib`. The notebook also saves fitted LabelEncoders to `encoders.joblib` so the demo accepts human-readable categories.

## Demo / GUI behavior
There are two ways to interact with the model via the demo UI (notebook Gradio cell or `app.py`):

- Single prediction (Interactive):
   - Input: Model, Year, Color, Engine.Cap.
   - Output: A Textbox showing the predicted price formatted with exactly two decimal places (e.g., `43281.21`). The notebook/app uses a formatted string to guarantee two-decimal display.

- Batch prediction (CSV upload):
   - Input: CSV with columns `Model, Year, Color, Engine.Cap` (the app also accepts `Engine Capacity` as an alias).
   - Output: A displayed DataFrame with columns: `Model, Year, Color, Engine Capacity, Predicted Price (RM)` (prices formatted with two decimals), and a downloadable CSV file containing the same formatted values.
   - There is also a Clear (✖) button that removes the table and the download link.

Notes about types:
- The UI displays prices as formatted strings (two decimal places) so values appear consistently in the frontend and downloaded CSV. If you need numeric outputs for further post-processing, we can keep a numeric rounded column in addition to the formatted string.

## Running the demo locally
1. Ensure `best_model.joblib` and `encoders.joblib` exist in the project root (run the notebook save-cell after training).
2. Install dependencies (see `requirements.txt`).

PowerShell example:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r .\requirements.txt
python .\app.py   # or run the Gradio cell in `main.ipynb`
```

## Requirements (notes)
- `requirements.txt` should pin `scikit-learn` to the version used to train the model to avoid joblib incompatibilities. A recommended minimal set includes:

```
gradio>=3.0,<4.0
scikit-learn==1.2.2    # pin this to your training version
pandas>=1.5.0
numpy>=1.23.0
joblib>=1.2.0
huggingface_hub>=0.13.0
scipy>=1.9.0
```

Adjust `scikit-learn==1.2.2` to the exact version you trained the model with.

## Deploying to Hugging Face Spaces (Gradio)
You can deploy this Gradio demo to Hugging Face Spaces. Two common approaches:

1) Commit artifacts to the Space repo
    - Add `best_model.joblib`, `encoders.joblib`, `app.py` (or leave the Gradio cell in `main.ipynb`), and `requirements.txt` to the Space repository.
    - From PowerShell:
       ```powershell
       git clone https://huggingface.co/spaces/<your-username>/<space-name>
       cd <space-name>
       copy ..\path\to\best_model.joblib .
       copy ..\path\to\encoders.joblib .
       copy ..\path\to\app.py .
       git add .
       git commit -m "Add demo app and model artifacts"
       git push
       ```
    - If you hit `authentication` errors when pushing, run `huggingface-cli login` (or `hf login`) and provide a token with repo write scope, or open the Space settings and add files through the web UI.

2) Download model artifacts from a Hugging Face model repo at runtime
    - If you prefer not to commit large model files into the Space repo, host `best_model.joblib` and `encoders.joblib` in a model repository on the Hub and download them at startup using `huggingface_hub` (the notebook/app includes a fallback loader).
    - Recommended env vars:
       - `HF_MODEL_REPO` — e.g. `username/my-car-model` (the model repo containing artifacts)
       - `HF_HUB_TOKEN` — if your model repo is private (set this in Space secrets or export locally)
    - Example snippet (already included in the app/notebook):
       ```python
       from huggingface_hub import hf_hub_download

       path = hf_hub_download(repo_id=os.environ['HF_MODEL_REPO'], filename='best_model.joblib', token=os.environ.get('HF_HUB_TOKEN'))
       ```

## Files
- `main.ipynb` — main notebook containing EDA, preprocessing, modeling and evaluation.
- `app.py` — Gradio demo for interactive predictions (single & batch).
- `requirements.txt` — Python packages required for the demo.
- `best_model.joblib`, `encoders.joblib` — model and encoder artifacts produced by the notebook (not committed by default unless you choose to).
- `Malaysia_Final_CarList_Compiled.csv` — dataset.

## Future Improvements
- Add feature importance visualization
- Experiment with XGBoost or CatBoost
- Add deployment automation (CI/CD)
- Improve app UI (add charts or value trends)
