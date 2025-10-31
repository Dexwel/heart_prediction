import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Dark theme / centered layout CSS
st.markdown(
    """
    <style>
    /* Page background and main container */
    .stApp, .main, .block-container {
        background-color: #000000 !important;
        color: #E6E6E6 !important;
    }
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    /* Make buttons and inputs slightly lighter for contrast */
    .stButton>button, .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        background-color: #0f0f0f !important;
        color: #E6E6E6 !important;
        border-color: #2b2b2b !important;
    }
    /* Keep links / markdown readable */
    a, p, label, span {
        color: #E6E6E6 !important;
    }
    /* Ensure separators are subtle */
    .stDivider, .stMarkdown hr {
        border-color: #222 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Heart Disease Prediction")
st.markdown("This app predicts heart disease")


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

# Centered input area using columns
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.header("Input values")
    st.write("Enter values for each feature. Categorical fields are pre-filled with observed values from the dataset (if available).")

    # Use a form so inputs are submitted together
    with st.form(key="input_form"):
        user_input = {}
        for feat in all_features:
            # If dataset available and feature appears non-numeric or has few unique values, show selectbox
            if hd is not None and feat in hd.columns:
                unique_vals = hd[feat].dropna().unique()
                # If feature is numeric but has few unique values, it's probably encoded categorical — show selectbox
                if (not pd.api.types.is_float_dtype(hd[feat]) and not pd.api.types.is_integer_dtype(hd[feat])) or len(unique_vals) <= 10:
                    options = list(np.unique(unique_vals))
                    try:
                        options = sorted(options)
                    except Exception:
                        options = [str(x) for x in options]
                    user_input[feat] = st.selectbox(feat, options, key=feat)
                    continue

            # fallback: numeric input for numerical features, otherwise text/number input
            if feat in numerical_features:
                # Prefer sliders for numeric features. If original dataset is available,
                # derive min/max/median from it. Otherwise fall back to a number input.
                try:
                    if hd is not None and feat in hd.columns and (pd.api.types.is_integer_dtype(hd[feat]) or pd.api.types.is_float_dtype(hd[feat])):
                        col_min = float(hd[feat].min())
                        col_max = float(hd[feat].max())
                        col_median = float(hd[feat].median())

                        # Expand zero-range columns to a reasonable window
                        if col_min == col_max:
                            span = abs(col_median) * 0.5 + 1.0
                            col_min = col_median - span
                            col_max = col_median + span

                        # Compute a sensible step; fall back to small default
                        raw_range = col_max - col_min
                        step = raw_range / 100.0 if raw_range > 0 else 1.0

                        # If the original column is integer-like, use integer slider
                        if pd.api.types.is_integer_dtype(hd[feat]):
                            imin = int(np.floor(col_min))
                            imax = int(np.ceil(col_max))
                            idefault = int(round(col_median))
                            user_input[feat] = st.slider(feat, imin, imax, idefault, key=feat)
                        else:
                            # Ensure step is a positive float
                            if step <= 0:
                                step = 0.1
                            user_input[feat] = st.slider(feat, float(col_min), float(col_max), float(col_median), step=step, key=feat)
                    else:
                        # No dataset context: fall back to number input with default 0.0
                        user_input[feat] = st.number_input(feat, value=float(0.0), key=feat)
                except Exception:
                    # Any unexpected issue: fallback to number input
                    user_input[feat] = st.number_input(feat, value=float(0.0), key=feat)
            else:
                # if hd present and column numeric, set default to median
                default = ""
                if hd is not None and feat in hd.columns and (pd.api.types.is_integer_dtype(hd[feat]) or pd.api.types.is_float_dtype(hd[feat])):
                    default = float(hd[feat].median())
                    user_input[feat] = st.number_input(feat, value=float(default), key=feat)
                else:
                    user_input[feat] = st.text_input(feat, value=default, key=feat)

        submitted = st.form_submit_button("Predict")

    if submitted:
        X = pd.DataFrame([user_input], columns=all_features)

        # Align dtypes with training dataset when possible
        if hd is not None:
            for col in all_features:
                if col in hd.columns:
                    try:
                        dtype = hd[col].dtype
                        if col in numerical_features:
                            X[col] = X[col].astype(float)
                        else:
                            if pd.api.types.is_integer_dtype(dtype):
                                X[col] = X[col].astype(int)
                            else:
                                X[col] = X[col]
                    except Exception:
                        pass
        else:
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
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)[0]
                try:
                    class_index = list(model.classes_).index(1)
                except ValueError:
                    class_index = 1 if len(probs) > 1 else 0
                prob = probs[class_index]
            elif hasattr(model, 'decision_function'):
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
