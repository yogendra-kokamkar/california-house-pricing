import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

# Streamlit app
st.title('California House Price Prediction')

# Input fields
MedInc = st.number_input('MedInc', min_value=0.0, format="%.2f")
HouseAge = st.number_input('HouseAge', min_value=0.0, format="%.2f")
AveRooms = st.number_input('AveRooms', min_value=0.0, format="%.2f")
AveBedrms = st.number_input('AveBedrms', min_value=0.0, format="%.2f")
Population = st.number_input('Population', min_value=0.0, format="%.2f")
AveOccup = st.number_input('AveOccup', min_value=0.0, format="%.2f")
Latitude = st.number_input('Latitude', min_value=-90.0, max_value=90.0, format="%.2f")
Longitude = st.number_input('Longitude', min_value=-180.0, max_value=180.0, format="%.2f")

# Prepare the input for the model
input_features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])

# Transform the input features
transformed_features = scalar.transform(input_features)

# Predict button
if st.button('Predict'):
    # Make the prediction
    prediction = regmodel.predict(transformed_features)
    
    # Display the result
    st.write(f'The House price prediction is ${prediction[0]:,.2f}')
