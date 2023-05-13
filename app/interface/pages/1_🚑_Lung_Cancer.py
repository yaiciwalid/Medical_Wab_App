import streamlit as st
import os
from PIL import Image
from fpdf import FPDF
import base64
import random
from app.models.ModelLungCancer import LungCancerModel


def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'


lung_model = LungCancerModel(load=True)


person = {
        'Age': 45,
        'Gender': 1,
        'Air_Pollution': 0,
        'Alcohol_use': 0,
        'Dust_Allergy': 0,
        'OccuPational_Hazards': 0,
        'Genetic_Risk': 0,
        'chronic_Lung_Disease': 0,
        'Balanced_Diet': 0,
        'Obesity': 0,
        'Smoking': 0,
        'Passive_Smoker': 0,
        'Chest_Pain': 0,
        'Coughing_of_Blood': 0,
        'Fatigue': 0,
        'Weight_Loss': 0,
        'Shortness_of_Breath': 0,
        'Wheezing': 0,
        'Swallowing_Difficulty': 0,
        'Clubbing_of_Finger_Nails': 0,
        'Frequent_Cold': 0,
        'Dry_Cough': 0,
        'Snoring': 0,
}

_input_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
_image_filepath = os.path.join(_input_path, "images", "lungs.jpeg")

image = Image.open(_image_filepath)
st.image(image)
col1, col2 = st.columns(2)

person['Age'] = col1.number_input('Age')
person['Gender'] = col2.number_input('Gender')
person['Air_Pollution'] = col1.number_input('Air_Pollution')
person['Alcohol_use'] = col2.number_input('Alcohol_use')
person['Dust_Allergy'] = col1.number_input('Dust_Allergy')
person['OccuPational_Hazards'] = col2.number_input('OccuPational_Hazards')
person['Genetic_Risk'] = col1.number_input('Genetic_Risk')
person['chronic_Lung_Disease'] = col2.number_input('chronic_Lung_Disease')
person['Balanced_Diet'] = col1.number_input('Balanced_Diet')
person['Obesity'] = col2.number_input('Obesity')
person['Smoking'] = col1.number_input('Smoking')
person['Passive_Smoker'] = col2.number_input('Passive_Smoker')
person['Chest_Pain'] = col1.number_input('Chest_Pain')
person['Coughing_of_Blood'] = col2.number_input('Coughing_of_Blood')
person['Fatigue'] = col1.number_input('Fatigue')
person['Weight_Loss'] = col2.number_input('Weight_Loss')
person['Shortness_of_Breath'] = col1.number_input('Shortness_of_Breath')
person['Wheezing'] = col2.number_input('Wheezing')
person['Swallowing_Difficulty'] = col1.number_input('Swallowing_Difficulty')
person['Clubbing_of_Finger_Nails'] = col2.number_input('Clubbing_of_Finger_Nails')
person['Frequent_Cold'] = col1.number_input('Frequent_Cold')
person['Dry_Cough'] = col2.number_input('Dry_Cough')
person['Snoring'] = col1.number_input('Snoring')

pressed = st.button("Generate Report Results")
if pressed:
    prediction = lung_model.predict(person=person)
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
        pdf.cell(60, 10, "Results : Patient positive to have Lung Cancer", "C")
    else:
        pdf.cell(60, 10, "Results : Patient Negative to Lung Cancer", "C")

    html = create_download_link(pdf.output(dest="S").encode("latin-1"), "lung_cancer_report_"+str(report_number))

    st.markdown(html, unsafe_allow_html=True)
