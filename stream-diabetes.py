import pickle
import streamlit as st
import pandas as pd

# Load model (pipeline Random Forest)
model = pickle.load(open('diabetes_model (2).sav', 'rb'))

# Judul
st.title('🩺 Prediksi Diabetes')
st.write("Masukkan data kesehatan untuk memprediksi kemungkinan diabetes")

# INPUT USER
col1, col2, col3, col4 = st.columns(4)

with col1:
    Pregnancies = st.number_input('Kehamilan', 0, 20)
with col2:
    Glucose = st.number_input('Glukosa', 50, 300)
with col3:
    BloodPressure = st.number_input('Tekanan Darah', 40, 150)
with col4:
    Age = st.number_input('Usia', 10, 100)

col5, col6, col7, col8 = st.columns(4)

with col5:
    Insulin = st.number_input('Insulin', 0, 300)
with col6:
    BMI = st.number_input('BMI', 10.0, 60.0)
with col7:
    DiabetesPedigreeFunction = st.number_input('Riwayat Diabetes', 0.0, 2.5)
with col8:
    SkinThickness = st.number_input('Ketebalan Kulit', 0, 100)

# PREDIKSI
if st.button('🔍 Prediksi'):
    input_data = pd.DataFrame(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness,
          Insulin, BMI, DiabetesPedigreeFunction, Age]],
        columns=[
            'Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age'
        ]
    )

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error('⚠️ Pasien berisiko Diabetes')
    else:
        st.success('✅ Pasien tidak berisiko Diabetes')

    st.info("Catatan: Ini hanya prediksi berbasis machine learning, bukan diagnosis medis.")
