import streamlit as st
import os
from PIL import Image
from fpdf import FPDF
import base64
import random
from app.models.Modeldiabetes import DiabeteModel


def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'


diabete_model = DiabeteModel(load=True)
diabete_model.pre_process(load=True)
diabete_model.split_train_data(0.4)
person = {
        'cholesterol_level': 164,
        'glucose_level': 93,
        'hdl_cholesterol': 59,
        'cholesterol_hdl_ratio': 2.4,
        'age': 96,
        'gender': "female",
        'height': 62,
        'weight': 217,
        'body_mass_idx': 0,
        'systolic_blood_pressure': 158,
        'diastolic_blood_pressure': 81,
        'waist_size': 50, 'hip_size': 50,
        'waist_hip_size_ratio': 1,
}

_input_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
_image_filepath = os.path.join(_input_path, "images", "diabeties.png")

image = Image.open(_image_filepath)
st.image(image)

col1, col2 = st.columns(2)

person['cholesterol_level'] = col1.number_input('cholesterol_level')
person['glucose_level'] = col2.number_input('glucose_level')
person['hdl_cholesterol'] = col1.number_input('hdl_cholesterol')
person['cholesterol_hdl_ratio'] = col2.number_input('cholesterol_hdl_ratio')
person['age'] = col1.number_input('age')
person['gender'] = col2.selectbox('gender', options=["female", "male"])
person['height'] = col1.number_input('height')
person['weight'] = col2.number_input('weight')
person['body_mass_idx'] = col1.number_input('body_mass_idx')
person['systolic_blood_pressure'] = col2.number_input('systolic_blood_pressure')
person['diastolic_blood_pressure'] = col1.number_input('diastolic_blood_pressure')
person['waist_size'] = col2.number_input('waist_size')
person['hip_size'] = col1.number_input('hip_size')
person['waist_hip_size_ratio'] = col2.number_input('waist_hip_size_ratio')

pressed = st.button("Generate Report Results")
if pressed:
    prediction = diabete_model.predict(person=person)
    report_number = random.randint(2_455_789_123, 2_999_789_123)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, 'Paris City Hospitals', align='C')
    pdf.ln()
    pdf.cell(60, 10, "Patient report "+str(report_number), "C")
    pdf.set_font('Arial', size=11)
    for k in person:
        pdf.ln()
        pdf.cell(60, 10, k+" : "+str(person[k]), "C")
    pdf.ln()
    pdf.set_font('Arial', 'B', 16)
    if (prediction == 1):
        pdf.cell(60, 10, "Results : Patient positive to have Diabetes", "C")
    else:
        pdf.cell(60, 10, "Results : Patient Negative to Diabetes", "C")

    html = create_download_link(pdf.output(dest="S").encode("latin-1"), "Diabetes_report_"+str(report_number))

    st.markdown(html, unsafe_allow_html=True)
