import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("Cancer Prediction Model")
st.write("This app predicts whether a tumor is **Malignant (M)** or **Benign (B)** based on user input.")

# Sidebar for input features
st.sidebar.header("Enter Tumor Characteristics:")

# Feature columns and their value ranges
columns = {
    "radius_mean": (5.0, 30.0),
    "texture_mean": (5.0, 40.0),
    "perimeter_mean": (40.0, 200.0),
    "area_mean": (150.0, 2500.0),
    "smoothness_mean": (0.05, 0.15),
    "compactness_mean": (0.01, 0.4),
    "concavity_mean": (0.0, 0.5),
    "concave points_mean": (0.0, 0.2),
    "symmetry_mean": (0.1, 0.3),
    "fractal_dimension_mean": (0.05, 0.1),
    "radius_se": (0.1, 5.0),
    "texture_se": (0.1, 5.0),
    "perimeter_se": (0.5, 20.0),
    "area_se": (5.0, 300.0),
    "smoothness_se": (0.001, 0.02),
    "compactness_se": (0.002, 0.15),
    "concavity_se": (0.0, 0.3),
    "concave points_se": (0.0, 0.05),
    "symmetry_se": (0.01, 0.08),
    "fractal_dimension_se": (0.001, 0.03),
    "radius_worst": (7.0, 40.0),
    "texture_worst": (7.0, 50.0),
    "perimeter_worst": (50.0, 250.0),
    "area_worst": (200.0, 3500.0),
    "smoothness_worst": (0.05, 0.2),
    "compactness_worst": (0.02, 1.5),
    "concavity_worst": (0.0, 1.5),
    "concave points_worst": (0.0, 0.3),
    "symmetry_worst": (0.1, 0.5),
    "fractal_dimension_worst": (0.05, 0.15)
}

# Gather user input for all features
input_data = []
for feature, (min_val, max_val) in columns.items():
    value = st.sidebar.slider(f"{feature}", min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
    input_data.append(value)

# Predict button
if st.button("Predict"):
    # Convert input data to a numpy array and scale it
    input_data = np.array(input_data).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)

    # Make predictions
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)

    # Map prediction result
    result = "Malignant (M)" if prediction[0] == 1 else "Benign (B)"

    # Display prediction result
    st.subheader(f"Prediction: {result}")

    # Display prediction probabilities
    st.write("### Prediction Probability:")
    proba_df = pd.DataFrame(prediction_proba, columns=["Benign (B)", "Malignant (M)"])
    st.bar_chart(proba_df.T)

    # Feature analysis graph
    st.write("### Feature Analysis:")
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(list(columns.keys()), input_data[0], color='skyblue')
    ax.set_xlabel("Value")
    ax.set_title("User Input Feature Values")
    st.pyplot(fig)

    # Radar chart visualization
    def plot_tumor_diagram(features, normalized_values):
        num_vars = len(features)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values = normalized_values + [normalized_values[0]]  # Close the circle
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color="skyblue", alpha=0.4)
        ax.plot(angles, values, color="blue", linewidth=2)
        ax.set_yticks([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, fontsize=10)
        ax.set_title("Tumor Feature Diagram", size=15, color="blue", y=1.1)
        return fig

    # Normalize values for radar chart
    max_values = [v[1] for v in columns.values()]
    normalized_values = [v / m for v, m in zip(input_data[0], max_values)]

    # Plot and display the radar chart
    st.write("### Tumor Visualization:")
    radar_fig = plot_tumor_diagram(list(columns.keys()), normalized_values)
    st.pyplot(radar_fig)