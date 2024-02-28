import streamlit as st
import numpy as np
import pandas as pd

st.header("Advertising Prediction App")

st.write("""
# Sales Prediction App

This app predicts the **Advertising Sales** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 0.7, 297, 100)
    Radio = st.sidebar.slider('Radio', 0, 50, 25)
    Newspaper = st.sidebar.slider('Newspaper', 0.3, 114, 70)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("Advertisingmodel.h5", "rb"))

prediction = loaded_model.predict(df)

st.subheader('Prediction')
st.write(prediction)

st.write(pd.DataFrame({
    'Intplan': ['yes', 'yes', 'yes', 'no'],
    'Churn Status': [0, 0, 0, 1]
}))
