#🧠 Cancer Prediction Model - Streamlit App
This project is a web-based interactive application built using Streamlit that predicts whether a tumor is Malignant (M) or Benign (B) using a pre-trained machine learning model. The prediction is based on user-input features derived from breast cancer diagnostic measurements.

#🚀 Features
🔬 Predicts Malignant (M) or Benign (B) tumors.

📊 Displays prediction probabilities.

📈 Visualizes input features using:

Horizontal bar chart

Radar chart (Tumor Feature Diagram)

🎛️ Interactive sliders for all 30 key tumor-related features.

#🧰 Files Included
app.py – Main Streamlit app script

model.pkl – Trained classification model (pickle format)

scaler.pkl – Feature scaler for preprocessing input (pickle format)

#🧠 Input Features
The model uses the following 30 features related to tumor characteristics:

Mean values: radius_mean, texture_mean, ..., fractal_dimension_mean

Standard error: radius_se, texture_se, ..., fractal_dimension_se

Worst values: radius_worst, texture_worst, ..., fractal_dimension_worst

