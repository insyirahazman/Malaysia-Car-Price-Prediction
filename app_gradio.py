import os
import joblib
import pandas as pd
import numpy as np
import gradio as gr

# Feature columns used by the notebook's final comparison (adjust if you changed features)
FEATURE_COLS = ['Model', 'Year', 'Color', 'Engine.Cap']


def load_model(path='best_model.joblib'):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file '{path}' not found. Run the notebook cell that saves the best model to 'best_model.joblib' and ensure it's in the project root."
        )
    return joblib.load(path)


try:
    model = load_model()
except Exception as exc:
    model = None
    load_error = str(exc)
else:
    load_error = None

# Try to load encoders (saved from the notebook) if available
encoders = None
enc_load_error = None
if os.path.exists('encoders.joblib'):
    try:
        encoders = joblib.load('encoders.joblib')
    except Exception as e:
        enc_load_error = str(e)


def predict_price(model_name: str, year: float, color: str, engine_cap: float):
    """
    Simple wrapper to predict price for a single example.
    NOTE: Categorical columns must be encoded the same way as during training (LabelEncoder mapping).
    If you used different features, update FEATURE_COLS accordingly.
    """
    if model is None:
        return f"Model not loaded: {load_error}"

    # Encode model name if we have encoders
    model_encoded = None
    if encoders is not None and 'Model' in encoders:
        try:
            model_encoded = encoders['Model'].transform([model_name])[0]
        except Exception:
            return f"Unknown model '{model_name}'. Allowed values: {list(encoders['Model'].classes_)}"
    else:
        # If no encoder, try to parse as numeric (less likely for model)
        try:
            model_encoded = float(model_name)
        except Exception:
            return "No encoder available for 'Model' and provided value is not numeric. Save encoders.joblib from the notebook."

    # Encode color if we have encoders and the color is given as a string
    color_encoded = None
    if encoders is not None and 'Color' in encoders:
        try:
            # transform expects array-like
            color_encoded = encoders['Color'].transform([color])[0]
        except Exception:
            # maybe user passed a numeric string; try to coerce
            try:
                color_encoded = float(color)
            except Exception:
                allowed = list(encoders['Color'].classes_)
                return f"Unknown color '{color}'. Allowed values: {allowed}"
    else:
        # No encoder available; try to parse as numeric
        try:
            color_encoded = float(color)
        except Exception:
            return "No encoder available and color value is not numeric. Save encoders.joblib from the notebook."

    X = pd.DataFrame([[model_encoded, year, color_encoded, engine_cap]], columns=FEATURE_COLS)
    pred = model.predict(X)[0]
    return float(np.round(pred, 2))


def batch_predict(csv_file):
    """Accepts a CSV file with the same feature columns and returns predictions appended."""
    if model is None:
        return f"Model not loaded: {load_error}"

    df = pd.read_csv(csv_file.name)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        return f"Uploaded CSV is missing required columns: {missing}"

    # If Color column contains strings and we have an encoder, transform it
    if 'Color' in df.columns and df['Color'].dtype == object:
        if encoders is None or 'Color' not in encoders:
            return "Uploaded CSV contains string Color values but no `encoders.joblib` was found. Save encoders from the notebook."
        # check for unknown labels
        unknowns = set(df['Color'][~df['Color'].isin(encoders['Color'].classes_)])
        if len(unknowns) > 0:
            return f"Uploaded CSV contains unknown Color labels: {list(unknowns)}. Allowed: {list(encoders['Color'].classes_)}"
        # safe to transform
        df['Color'] = encoders['Color'].transform(df['Color'])

    # If Model column contains strings and we have an encoder, transform it
    if 'Model' in df.columns and df['Model'].dtype == object:
        if encoders is None or 'Model' not in encoders:
            return "Uploaded CSV contains string Model values but no `encoders.joblib` was found. Save encoders from the notebook."
        unknowns = set(df['Model'][~df['Model'].isin(encoders['Model'].classes_)])
        if len(unknowns) > 0:
            return f"Uploaded CSV contains unknown Model labels: {list(unknowns)}. Allowed: {list(encoders['Model'].classes_)}"
        df['Model'] = encoders['Model'].transform(df['Model'])

    preds = model.predict(df[FEATURE_COLS])
    df['predicted_price'] = np.round(preds, 2)
    return df


title = "Malaysia Car Price Predictor"
description = (
    "Enter the features used by the model or upload a CSV containing the same columns (Model, Year, Color, Engine.Cap).\n"
)

iface_single = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Textbox(label='Model', value=''),
        gr.Number(label='Year', value=2016),
        gr.Textbox(label='Color (name or encoded)', value=''),
        gr.Number(label='Engine.Cap', value=1600),
    ],
    outputs=gr.Number(label='Predicted Price (RM)'),
    title=title,
    description=description,
)

def main():
    # Launch single-prediction demo (no tabs)
    iface_single.launch(server_name='0.0.0.0')


if __name__ == '__main__':
    main()
