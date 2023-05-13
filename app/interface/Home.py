import streamlit as st
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

st.write('# Welcome Doctor ! 👨‍⚕️👩‍⚕️')

st.markdown("""
    This app is dedicated to you.
    We try to use advanced AI models in order to help you
    facilitate the analyzes 🩺 of your patients and also
    predict the chance of your patience to get some diseases 🤒
    in early stages and save many lives ❤️‍🩹 """)
