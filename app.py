import streamlit as st
import joblib
import numpy as np

# Load the trained KNN model
model = joblib.load("knn_model.pkl")

# Streamlit UI
st.title("🌱 KNN Model for Iris Classification")
st.write("Enter the flower's features to predict its class.")

# Input fields for user to enter feature values
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Predict button
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]

    class_names = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"🌼 The predicted class is: {class_names[prediction]}")

