import streamlit as st
import os
from PIL import Image
from fpdf import FPDF
import base64
import random
from app.models.Modelpneumonia.ModelPneumonia import PneumoniaModel


def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'


_input_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
_image_filepath = os.path.join(_input_path, "images", "pneumonia.png")

image = Image.open(_image_filepath)
st.image(image)

model = PneumoniaModel(load=True)
uploded_file = None
uploded_file = st.file_uploader("Upload Pneumonia X-Ray Image Files", type=['jpeg'])
st.warning("Only .jpeg file fomrats are accepted")

if uploded_file is not None:
    pressed = st.button("Generate Report Results")

    if pressed:
        prediction = model.predict(uploded_file)
        report_number = random.randint(2_455_789_123, 2_999_789_123)
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(200, 10, 'Paris City Hospitals', align='C')
        pdf.ln()
        pdf.cell(60, 10, "Patient report "+str(report_number), "C")
        pdf.ln()
        img = Image.open(uploded_file)    
        x, y = img.size
        w = img.width
        h = img.height
        img = img.resize((400, 400))
        img.save("pneum_image.jpeg")
        pdf.image("pneum_image.jpeg", type="jpeg")
        os.remove("pneum_image.jpeg")
        pdf.ln()
        pdf.set_font('Arial', 'B', 16)
        if (prediction == 1):
            pdf.cell(60, 10, "Results : Patient positive to have Pneumonia", "C")
        else:
            pdf.cell(60, 10, "Results : Patient Negative to Pneumonia", "C")

        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "Pneumonia_report_"+str(report_number))

        st.markdown(html, unsafe_allow_html=True)
