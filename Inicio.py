import streamlit as st
import numpy as np
import pandas as pd

from funciones import menu_pages, load_data_csv
# Page title
# configuration
st.set_page_config(
    page_title="VGCLAB",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ESTILOS
with open('style/style.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

image = 'images/vgclab-negro.jpg'

# --------------------------
# sidebar
# --------------------------
st.sidebar.image(image)

# pages
menu_pages()

st.sidebar.write("[Fundación Chile 21](https://chile21.cl/)")
st.sidebar.divider()
st.sidebar.write("San Sebastián 2807, Las Condes, Santiago de Chile")

st.header("Fundación Chile 21")
st.subheader("Plataforma de datos")
st.divider()

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    image = 'images/maceda.jpg'
    st.image(image)
with col2:
    st.write("El Laboratorio sobre Violencias y Gestión de Conflictos VGC LAB es un espacio de reflexión, investigación colaborativa y generación de propuestas de intervención social para abordar la violencia y los conflictos en diferentes ámbitos, con el fin de promover una cultura de la paz y buena gestión de los conflictos.")

with col3:
    image = 'images/maceda.jpg'
