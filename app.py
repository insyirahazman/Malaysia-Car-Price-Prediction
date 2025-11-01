import os
import joblib
import pandas as pd
import numpy as np
import tempfile
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


def safe_encode(col_name, val):
    """
    If encoders exist and contain col_name, try to transform val using the encoder.
    If encoder missing, attempt to cast to float and return numeric value.
    On unknown label, raise ValueError with informative message.
    """
    if encoders is not None and col_name in encoders:
        le = encoders[col_name]
        try:
            return le.transform([val])[0]
        except Exception:
            # unknown label
            raise ValueError(f"Unknown {col_name} '{val}'. Allowed values: {list(le.classes_)}")
    else:
        # no encoder: try numeric fallback
        try:
            return float(val)
        except Exception:
            raise ValueError(f"No encoder for '{col_name}' and provided value is not numeric. Provide numeric value or save encoders.joblib.")


def predict_price(model_name: str, year: float, color: str, engine_cap: float):
    """
    Simple wrapper to predict price for a single example.
    Returns a string formatted with two decimal places (e.g. '43281.21').
    """
    if model is None:
        return f"Model not loaded: {load_error}"

    try:
        model_encoded = safe_encode('Model', model_name)
        color_encoded = safe_encode('Color', color)
        engine_encoded = safe_encode('Engine.Cap', engine_cap)
    except ValueError as ve:
        return str(ve)

    # Year should be numeric; try to coerce
    try:
        year_val = float(year)
    except Exception:
        return "Year must be numeric."

    X = pd.DataFrame([[model_encoded, year_val, color_encoded, engine_encoded]], columns=FEATURE_COLS)
    pred = model.predict(X)[0]
    # Return formatted string with two decimals so the UI shows exactly two decimals
    return f"{pred:.2f}"


def batch_predict(csv_file):
    """Accepts a CSV file with the same feature columns and returns predictions appended.

    Output columns (display): Model, Year, Color, Engine Capacity, Price (Predicted)
    """
    if model is None:
        raise ValueError(f"Model not loaded: {load_error}")

    # Read uploaded CSV (gr.File gives .name)
    df = pd.read_csv(csv_file.name)

    # Accept either 'Engine.Cap' or 'Engine Capacity' as input; normalize to 'Engine.Cap'
    if 'Engine Capacity' in df.columns and 'Engine.Cap' not in df.columns:
        df['Engine.Cap'] = df['Engine Capacity']

    required = ['Model', 'Year', 'Color', 'Engine.Cap']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing required columns: {missing}")

    # Preserve original human-readable columns for display
    df['_orig_Model'] = df['Model'].astype(str)
    df['_orig_Color'] = df['Color'].astype(str)
    df['_orig_Engine_Cap'] = df['Engine.Cap']

    # Prepare encoded columns for prediction
    def prepare_encoding(col):
        enc_col = f"{col}_enc"
        if encoders is not None and col in encoders:
            # If column contains object (strings), check for unknowns and transform
            if df[col].dtype == object:
                unknowns = set(df[col][~df[col].isin(encoders[col].classes_)])
                if len(unknowns) > 0:
                    raise ValueError(f"Uploaded CSV contains unknown {col} labels: {list(unknowns)}. Allowed: {list(encoders[col].classes_)}")
                df[enc_col] = encoders[col].transform(df[col])
            else:
                try:
                    # try transforming stringified values
                    df[enc_col] = encoders[col].transform(df[col].astype(str))
                except Exception:
                    df[enc_col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[enc_col] = pd.to_numeric(df[col], errors='coerce')

    prepare_encoding('Model')
    prepare_encoding('Color')
    prepare_encoding('Engine.Cap')

    # Ensure Year is numeric
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    # Build the prediction DataFrame using the SAME feature names used during training.
    # This is the critical fix: rename encoded columns back to the original feature names.
    X_for_pred = pd.DataFrame({
        'Model': df['Model_enc'],
        'Year': df['Year'],
        'Color': df['Color_enc'],
        'Engine.Cap': df['Engine.Cap_enc']
    })

    if X_for_pred.isnull().any(axis=1).any():
        bad = X_for_pred.isnull().any(axis=1)
        raise ValueError("Uploaded CSV contains non-numeric or missing values in required columns after processing. Problem rows (sample):\n"
                         + df.loc[bad, ['Model', 'Year', 'Color', 'Engine.Cap']].head(5).to_csv(index=False))

    # Now call model.predict with the DataFrame that has the same column names as at fit time
    preds = model.predict(X_for_pred)
    # Format predicted prices as two-decimal strings to guarantee display (e.g. '43281.21')
    df['Predicted Price (RM)'] = [f"{v:.2f}" for v in preds]

    # Build the display DataFrame with requested column names/order
    # Keep human-readable Model and Color values
    out = pd.DataFrame({
        'Model': df['_orig_Model'],
        'Year': df['Year'].astype(int) if not df['Year'].isnull().any() else df['Year'],
        'Color': df['_orig_Color'],
        'Engine Capacity': df['_orig_Engine_Cap'],
        'Predicted Price (RM)': df['Predicted Price (RM)']
    })
    return out

def batch_predict_file(csv_file):
    """
    Run batch prediction and write results to a temporary CSV file.
    Returns the path to the generated CSV, suitable for Gradio File output.
    """
    if csv_file is None:
        raise ValueError("No CSV file provided for download.")
    # Reuse batch_predict to produce the DataFrame (this will raise informative errors if input invalid)
    out_df = batch_predict(csv_file)
    # Write to a temp file and return its path (Gradio accepts a filepath for File output)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    out_df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def batch_predict_and_file(csv_file):
    """
    Run batch prediction and return both the display DataFrame and a temp CSV path
    so a single Gradio action can update the Dataframe and provide a downloadable file.
    """
    if csv_file is None:
        raise ValueError("No CSV file provided for download.")

    # Produce DataFrame using existing batch_predict (this raises clear errors on invalid input)
    out_df = batch_predict(csv_file)

    # Write results to a temp CSV and return both outputs
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    out_df.to_csv(tmp.name, index=False)
    tmp.close()
    return out_df, tmp.name

def clear_batch_outputs():
    """
    Clear the batch outputs displayed in the UI.
    Returns an empty DataFrame for the table and None for the file output so Gradio clears both.
    """
    return pd.DataFrame(), None


# Prepare dropdown options based on encoders if available (fallbacks handled in UI creation)
model_choices = list(encoders['Model'].classes_) if encoders is not None and 'Model' in encoders else []
color_choices = list(encoders['Color'].classes_) if encoders is not None and 'Color' in encoders else []
engine_choices = list(encoders['Engine.Cap'].classes_) if encoders is not None and 'Engine.Cap' in encoders else [] 
engine_choices = list(encoders['Engine.Cap'].classes_) if encoders is not None and 'Engine.Cap' in encoders else []

title = "Perodua Car Price Predictor"
description = "Single prediction or upload a CSV for batch predictions."

# Build Gradio app using Blocks with Tabs
with gr.Blocks() as demo:
    gr.Markdown(f"## {title}\n\n{description}")

    with gr.Tab("Single"):
        # Model input: dropdown if encoder present, otherwise textbox
        if model_choices:
            model_input = gr.Dropdown(choices=model_choices, label="Model", value=model_choices[0])
        else:
            model_input = gr.Textbox(label="Model (no encoders found; enter encoded numeric or save encoders)", value="")

        year_input = gr.Number(label="Year", value=2016)

        # Color input
        if color_choices:
            color_input = gr.Dropdown(choices=color_choices, label="Color", value=color_choices[0])
        else:
            color_input = gr.Textbox(label="Color (no encoders found; enter encoded numeric or save encoders)", value="")

        # Engine Capacity: if encoder exists, use dropdown; else use Number input
        if engine_choices:
            engine_input = gr.Dropdown(choices=engine_choices, label="Engine Capacity", value=engine_choices[0])
        else:
            engine_input = gr.Number(label="Engine Capacity", value=1600)

        predict_btn = gr.Button("Predict")
        # Use a Textbox to display formatted price (exactly two decimals)
        single_out = gr.Textbox(label="Predicted Price (RM)")

        predict_btn.click(fn=predict_price, inputs=[model_input, year_input, color_input, engine_input], outputs=single_out)

    with gr.Tab("Batch"):
        gr.Markdown("Upload CSV with columns: Model, Year, Color, Engine.Cap (or 'Engine Capacity').")
        csv_input = gr.File(label="Upload CSV", file_types=['.csv'])

        # Predict -> show table and generate downloadable CSV
        batch_btn = gr.Button("Predict & Prepare Download")
        df_out = gr.Dataframe(headers=None)
        file_out = gr.File(label="Download Predictions (CSV)")

        # Single click will update both the table and provide a downloadable CSV
        batch_btn.click(fn=batch_predict_and_file, inputs=csv_input, outputs=[df_out, file_out])

        # Clear (✖) button — clears the table and removes the file link
        clear_btn = gr.Button("✖ Clear Output", variant="secondary")
        clear_btn.click(fn=clear_batch_outputs, inputs=[], outputs=[df_out, file_out])

# Launch app
def main():
    demo.launch(share=True)


if __name__ == '__main__':
    main()