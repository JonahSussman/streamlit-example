import numpy as np 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image


# Load  model a 
model = joblib.load(open("model-final.joblib","rb"))

def data_preprocessor(df):
    """this function preprocess the user input
        return type: pandas dataframe
    """
    df_rename = {
        'Pregnancies': 'pregnancies',
        'Glucose': 'glucose',
        'BloodPressure': 'blood_pressure',
        'SkinThickness': 'skin_thickness',
        'Insulin': 'insulin',
        'BMI': 'bmi',
        'DiabetesPedigreeFunction': 'pedigree_function',
        'Age': 'age',
        'Outcome': 'outcome'
        }
    df.rename(columns=df_rename, inplace=True)
    df.head()
    return df

def visualize_confidence_level(prediction_proba):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
    """
    data = (prediction_proba[0]*100).round(2)
    fig, ax = plt.subplots()
    ax.pie(data, labels=['No diabetes', 'Diabetes'], autopct='%1.1f%%')

    st.pyplot(fig)
    return

st.write("""
# Diabetes Prediction Data Science Web-App 
This app predicts the potentiality of a high-risk diabetes patient using **features** input via the **side panel** 
""")

image = Image.open('insulin-image.gif')
st.image(image, caption='Chemical image of insulin B', use_column_width=True)

st.sidebar.header('User Input Parameters') #user input parameter collection with streamlit side bar


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe

    """
    pregnancies       = st.sidebar.slider('pregnancies', 0.0, 20.0, 0.0)
    glucose           = st.sidebar.slider('glucose', 25.0, 200.0, 100.0)
    blood_pressure    = st.sidebar.slider('blood_pressure', 20.0, 140.0, 60.0)
    skin_thickness    = st.sidebar.slider('skin_thickness', 0.0, 100.0, 20.0)
    insulin           = st.sidebar.slider('insulin', 0.0, 400.0, 50.0)
    bmi               = st.sidebar.slider('bmi', 10.0, 70.0, 21.0)
    pedigree_function = st.sidebar.slider('pedigree_function', 0.0, 2.5, 0.2)
    age               = st.sidebar.slider('age', 21.0, 90.0, 21.0)
    
    features = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'bmi': bmi,
        'pedigree_function': pedigree_function,
        'age': age
        }
    data = pd.DataFrame(features,index=[0])

    return data

user_input_df = get_user_input()
processed_user_input = data_preprocessor(user_input_df)

st.subheader('User Input parameters')
st.write(user_input_df)

prediction = model.predict(processed_user_input)
prediction_proba = model.predict_proba(processed_user_input)

visualize_confidence_level(prediction_proba)