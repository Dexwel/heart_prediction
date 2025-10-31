import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("Heart Disease Prediction")

st.markdown(
    "This app predicts heart disease")


@st.cache_data
def load_pickle(path="best_model.pkl"):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model_data = load_pickle()
if model_data is None:
    st.error("Could not find 'best_model.pkl'. Run the notebook cell that saves the tuned model to create it.")
    st.stop()

model = model_data['model']
scaler = model_data['scaler']
numerical_features = model_data['numerical_features']
all_features = model_data['all_features']

# Try to read the original dataset to infer categorical options
try:
    hd = pd.read_csv('data.csv')
except Exception:
    hd = None

st.sidebar.header("Input values")
st.sidebar.write("Enter values for each feature. Categorical fields are pre-filled with observed values from the dataset (if available).")

user_input = {}
for feat in all_features:
    # If dataset available and feature appears non-numeric or has few unique values, show selectbox
    if hd is not None and feat in hd.columns:
        unique_vals = hd[feat].dropna().unique()
        # If feature is numeric but has few unique values, it's probably encoded categorical — show selectbox
        if (not pd.api.types.is_float_dtype(hd[feat]) and not pd.api.types.is_integer_dtype(hd[feat])) or len(unique_vals) <= 10:
            options = list(np.unique(unique_vals))
            # sort but keep numbers as numbers
            try:
                options = sorted(options)
            except Exception:
                options = [str(x) for x in options]
            user_input[feat] = st.sidebar.selectbox(feat, options)
            continue

    # fallback: numeric input for numerical features, otherwise number_input
    if feat in numerical_features:
        user_input[feat] = st.sidebar.number_input(feat, value=float(0.0))
    else:
        # if hd present and column numeric, set default to median
        default = 0.0
        if hd is not None and feat in hd.columns and (pd.api.types.is_integer_dtype(hd[feat]) or pd.api.types.is_float_dtype(hd[feat])):
            default = float(hd[feat].median())
        user_input[feat] = st.sidebar.number_input(feat, value=default)


if st.sidebar.button("Predict"):
    X = pd.DataFrame([user_input], columns=all_features)

    # Align dtypes with training dataset when possible
    if hd is not None:
        for col in all_features:
            if col in hd.columns:
                try:
                    # coerce to the same dtype as the original dataset column
                    dtype = hd[col].dtype
                    # For numeric columns, convert to float to match scaler expectation
                    if col in numerical_features:
                        X[col] = X[col].astype(float)
                    else:
                        # If original dtype is integer-like, try to convert
                        if pd.api.types.is_integer_dtype(dtype):
                            X[col] = X[col].astype(int)
                        else:
                            # leave object/str types as-is
                            X[col] = X[col]
                except Exception:
                    # If conversion fails, keep original and handle later
                    pass
    else:
        # If we don't have the original dataset, at least ensure numeric features are floats
        try:
            X[numerical_features] = X[numerical_features].astype(float)
        except Exception:
            st.error("Could not convert numerical inputs to floats. Check your entries.")
            st.stop()

    # Check for missing values
    if X.isnull().any().any():
        missing = X.columns[X.isnull().any()].tolist()
        st.error(f"Missing values detected for features: {missing}. Please fill all fields.")
        st.stop()

    # Scale numeric features
    try:
        X[numerical_features] = scaler.transform(X[numerical_features])
    except Exception as e:
        st.error(f"Error applying scaler to numeric features: {e}")
        st.stop()

    try:
        prediction = model.predict(X)[0]
        prob = None
        # Extract probability for class '1' robustly using model.classes_
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[0]
            try:
                class_index = list(model.classes_).index(1)
            except ValueError:
                # If class '1' not found, default to the second column if present
                class_index = 1 if len(probs) > 1 else 0
            prob = probs[class_index]
        elif hasattr(model, 'decision_function'):
            # Map decision function to a pseudo-probability via sigmoid (not calibrated)
            score = model.decision_function(X)[0]
            prob = 1 / (1 + np.exp(-score))

        st.write("## Prediction")
        if int(prediction) == 1:
            st.error("Model prediction: Heart Disease (1)")
        else:
            st.success("Model prediction: No Heart Disease (0)")

        if prob is not None:
            st.write(f"Predicted probability of heart disease: {prob:.3f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---\nThis Machine Learning application is built with ❤️ by Daniel Adediran")
