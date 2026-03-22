import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🚗 Car Price Prediction")

engine_size = st.number_input("Engine Size", value=100)
horsepower = st.number_input("Horsepower", value=100)

if st.button("Predict"):
    
    # Get number of features expected
    n_features = scaler.n_features_in_
    
    # Create empty input
    data = np.zeros((1, n_features))
    
    # Fill first 2 features (approx mapping)
    data[0][0] = engine_size
    data[0][1] = horsepower

    data = scaler.transform(data)
    prediction = model.predict(data)

    st.success(f"Predicted Price: {prediction[0]:,.2f}")