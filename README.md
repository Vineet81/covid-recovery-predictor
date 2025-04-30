# DEMO Link: https://covid-recovery-predictor-hldtqkycssea4xv4khzsmu.streamlit.app/
# 🦠 COVID Recovery Risk Predictor App (Streamlit)

This is a Streamlit web application that predicts the **recovery risk status** of a COVID-19 patient based on multiple health indicators such as age, vaccination status, liver function, GFR, spirometry, immunoglobulin, and T-cell count.

The model was built using **synthetic data**, so some outputs may not reflect realistic clinical predictions. However, this project demonstrates the complete ML pipeline: from data preprocessing, model training, and evaluation, to building a clean UI using Streamlit.

---

## 🎯 Prediction Goal

**Target variable**: `Recovery_Status`  
Possible outcomes:

- `Recovered`
- `Slight_Risk`
- `Mild_Risk`
- `Severe_Risk`

**Input features**:

- `AGE`: Age in years (18–80)
- `VACCINATION`: 0 = Not vaccinated, 1 = Vaccinated
- `LIVER`: 0 = Abnormal, 1 = Normal
- `GFR`: 0 = Abnormal, 1 = Normal
- `IMMUNOGLOBULIN`: 0 = Abnormal, 1 = Normal
- `SPIROMETRY`: 0 = Abnormal, 1 = Normal
- `T_CELL_COUNT`: 0 = Abnormal, 1 = Normal

---

## 🧪 Model Used

- `RandomForestClassifier` from Scikit-learn
- Trained on a **synthetic dataset**
- Saved and loaded using `joblib`
- Deployed using **Streamlit**

---

## 🖥️ Features

- Clean, responsive user interface using Streamlit
- Takes user input using sliders and dropdowns
- Displays predicted recovery status instantly
- Ready for future integration with real-world health data

## 📁 Project Structure

COVID-RECOVERY-APP/
├── app.py # Streamlit app
├── covid_recovery_model.pkl # Trained classification model (joblib)
├── requirements.txt # Dependencies
├── Health_data.csv # Synthetic dataset
├── Screenshots/ # App UI screenshots
└── README.md # Project documentation

## Instructions:

# 1.Create and activate a virtual environment

python -m venv venv
source venv/bin/activate (Linux/mac) # venv\Scripts\activate (windows)

# 2.Install required packages

pip install -r requirements.txt

# 3. Run the Streamlit app

streamlit run app.py
