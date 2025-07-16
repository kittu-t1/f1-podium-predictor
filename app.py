import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load('xgboost_f1_model.pkl')

st.title("🏎️ F1 Podium Predictor")
uploaded_file = st.file_uploader("Upload CSV", type="csv")
st.markdown("Expected columns: `grid`, `age_at_race`, `constructor_encoded`, `year`")

# Everything inside this block:
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data", data.head())

    if st.button("Predict"):
        # Predictions
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]  # prob of podium class

        data['Podium_Predicted'] = predictions
        data['Podium_Probability (%)'] = (probabilities * 100).round(2)

        st.write("### Results with Confidence", data)
        st.success("✅ Predictions include model confidence.")

        # Feature Importance
st.write("### 🔍 Feature Importance (from XGBoost)")

# Use only model input feature names
features = ['grid', 'age_at_race', 'constructor_encoded', 'year']
importance = model.feature_importances_

importance_df = pd.Series(importance, index=features).sort_values()

plt.figure(figsize=(8, 4))
importance_df.plot(kind='barh')
plt.xlabel("Importance Score")
plt.title("XGBoost - Feature Importance")
st.pyplot(plt)

