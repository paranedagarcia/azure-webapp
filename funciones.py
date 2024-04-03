import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data
def load_data_csv(filename, chunks=False):
    # Load the data from the CSV file separated by comma or semicomma

    if filename is not None:
        try:
            data = pd.read_csv(filename, sep=",",
                               low_memory=False, index_col=0)
        except:
            data = pd.read_csv(filename, sep=";",
                               low_memory=False, index_col=0)
        return data


def menu_inicio():
    st.sidebar.page_link("Inicio.py", label="Inicio")


@st.cache_data
def menu_pages():
    st.sidebar.page_link("Inicio.py", label=":house: Inicio")
    st.sidebar.page_link("pages/Conflictos_mapuches.py",
                         label=":bar_chart: Conflictos mapuches")
    st.sidebar.page_link("pages/Conflictos_sociales.py",
                         label=":bar_chart: Conflictos sociales")
    st.sidebar.page_link("pages/Detenciones_por_delitos.py",
                         label=":bar_chart: Detenciones por delitos")
    st.sidebar.page_link("pages/Salud_mental.py",
                         label=":bar_chart: Salud mental")
