import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained pipeline with preprocessing
model = joblib.load("covid_recovery_model.pkl")

st.set_page_config(page_title="COVID-19 Recovery Dashboard", layout="wide")
st.title("🦠 COVID-19 Recovery Risk Predictor Dashboard")

# Sidebar for navigation
st.sidebar.header("Navigation")
options = st.sidebar.radio("Choose Action", ["Single Prediction", "Bulk Prediction & Visualization"])

# -------------------- SINGLE PREDICTION --------------------
if options == "Single Prediction":
    st.subheader("🧍 Predict for One Patient")

    age = st.slider("Age", 18, 90, 35)

    vaccination = st.selectbox("Vaccination", ["YES", "NO"])
    liver = st.selectbox("Liver", ["Normal", "Abnormal"])
    gfr = st.selectbox("GFR (Kidney)", ["Normal", "Abnormal"])
    immuno = st.selectbox("Immunoglobulin", ["Normal", "Abnormal"])
    spirometry = st.selectbox("Spirometry", ["Normal", "Abnormal"])
    t_cell = st.selectbox("T Cell Count", ["Normal", "Abnormal"])

    # Create DataFrame
    single_input = pd.DataFrame([{
        "AGE": age,
        "VACCINATION": vaccination,
        "LIVER": liver,
        "GFR": gfr,
        "IMMUNOGLOBULIN": immuno,
        "SPIROMETRY": spirometry,
        "T_CELL_COUNT": t_cell
    }])

    if st.button("Predict"):
        prediction = model.predict(single_input)[0]
        st.success(f"🩺 Predicted Recovery Status: **{prediction}**")

# -------------------- BULK PREDICTION --------------------
elif options == "Bulk Prediction & Visualization":
    st.subheader("📤 Upload Patient Data for Bulk Prediction")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        try:
            preds = model.predict(df)
            df["Predicted_Recovery_Status"] = preds
            st.success("✅ Predictions Completed")
            st.dataframe(df)

            # Download
            download_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", download_csv, "predicted_results.csv", "text/csv")

            st.subheader("📊 Visualizations")

            # Pie Chart
            st.markdown("### 🥧 Recovery Status Distribution")
            pie_data = df["Predicted_Recovery_Status"].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
            ax1.axis("equal")
            st.pyplot(fig1)

            # Age vs Recovery
            st.markdown("### 📈 Age vs Recovery Risk")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.boxplot(data=df, x="Predicted_Recovery_Status", y="AGE", ax=ax2)
            st.pyplot(fig2)

            # Heatmap of Features
            st.markdown("### 🔥 Correlation Heatmap (Numerical)")
            numeric_df = df.select_dtypes(include='number')
            if not numeric_df.empty:
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax3)
                st.pyplot(fig3)
            else:
                st.info("ℹ️ No numeric columns to display correlation heatmap.")
        except Exception as e:
            st.error(f"❌ Prediction Failed: {e}")
