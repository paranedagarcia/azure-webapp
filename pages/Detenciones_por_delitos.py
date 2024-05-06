'''
Detenciones.
Fecha: 2024-01-12
Autor: Patricio Araneda
'''

# librerias
import os
import pandas as pd
import numpy as np
import time
import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.metric_cards import style_metric_cards

import base64
from io import BytesIO
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from millify import millify
import pygwalker as pgw
# from pandasai import SmartDatalake  # para multiples dataframes
from pandasai import Agent
from pandasai.llm.openai import OpenAI
from pandasai.responses.streamlit_response import StreamlitResponse

from datetime import datetime
from funciones import menu_pages, load_data_csv
from dotenv import load_dotenv

from funciones import load_data_csv

# configuration
st.set_page_config(
    page_title="Eficiencia de justicia",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ESTILOS
with open('style/style.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

load_dotenv()

API_KEY = os.environ['OPENAI_API_KEY']  # st.secrets['OPENAI_API_KEY']
openai_api_key = API_KEY

menu_pages()
progress_text = "Cargando datos..."
pro_bar = st.progress(0.15, text=progress_text)

comunas_file = "https://data.vgclab.cl/public_data/comunas.csv"
comunas = pd.read_csv(comunas_file, header=0, sep=";")
regiones = comunas["REGION"].unique().tolist()
provincias = comunas["PROVINCIA"].unique().tolist()

public = "https://vgclabfiles.blob.core.windows.net/public/"
datos = public+"dataset_detenciones.csv"
# datos = "data/dataset_detenciones.csv"
try:
    df = load_data_csv(datos)
# df = pd.read_csv("data/dataset_detenciones.csv", header=0, sep=",")
# Convert 'FECHA' to datetime type
except:
    st.error("Error al cargar los datos")
    st.stop()

pro_bar.progress(0.3, text="Estableciendo métricas...")

df['FECHA'] = pd.to_datetime(df['FECHA'], format='%Y-%m-%d')

# Extract year and create 'YEAR' column
df['YEAR'] = df['FECHA'].dt.year

delitos = df["DELITO"].unique().tolist()
delitos = [x for x in delitos if str(x) != 'nan']
delito_selection = st.sidebar.multiselect("Delitos:", delitos, [
                                          'Rb_Violencia_o_Intimidación', 'Rb_Sorpresa', 'Rb_Fuerza', 'Rb_Vehìculo', 'Hurtos'])
if delito_selection is not None:
    df = df[df["DELITO"].isin(delito_selection)]
else:
    df = df


# filter delitos
# selected_unidad = st.sidebar.selectbox(
#     "Seleccione agrupación", ["Ninguna", "Comuna", "Provincia", "Región"])

st.subheader("Eficiencia de la Justicia")
# st.subheader("Detenciones " + selected_unidad)

# --------------------------
# METRICAS
# --------------------------
delito_mayor = df.groupby('DELITO')['CASOS'].sum().nlargest(
    1).reset_index(name='max')

# st.write("Total de detenciones: ", df.shape[0])
detenciones_totales = df.shape[0]
delitos_totales = df["DELITO"].nunique()

# Calcula el total de casos para cada tipo de delito
delitos_suma = df.groupby('DELITO')['CASOS'].sum().reset_index()

# Calcula el total de casos para el cálculo del porcentaje
total_cases = delitos_suma['CASOS'].sum()


# Ordena los delitos por número de casos para obtener los delitos más comunes
# crime_counts_sorted = crime_counts.sort_values('CASOS', ascending=False)

tabPanel, tabTable, tabIA, tabInfo = st.tabs(
    ["Panel", "Tabla", "IA-EDA", "Información"])

with tabPanel:
    # st.write(delito_mayor.iloc[0, 0] + " es el delito más común con " +
    #         str(delito_mayor.iloc[0, 1]) + " casos.")
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    col1.metric("Total de detenciones", millify(
        detenciones_totales, precision=2))
    col2.metric("Tipo de delitos", delitos_totales)
    col3.metric(delito_mayor.iloc[0, 0], millify(
        delito_mayor.iloc[0, 1], precision=2))
    col4.metric("Comuna", "La Florida")

    style_metric_cards()

    pro_bar.progress(0.5, text="Construyendo gráficos...")

    det = df.groupby(["COMUNA", "DELITO", "YEAR"]).agg({'CASOS': sum})

    dfcomuna = df.groupby('COMUNA')['CASOS'].sum().nlargest(
        10).reset_index(name='suma')
    dfcomuna = dfcomuna.sort_values(by='suma', ascending=False)

    dfregion = df.groupby('REGION')['CASOS'].sum().reset_index(name='suma')
    dfregion = dfregion.sort_values(by='suma', ascending=False)

    # ------------------------------------------------
    # Tendencia de delitos a lo largo del tiempo

    df['FECHA'] = pd.to_datetime(df['FECHA'])
    delitos_por_fecha = df.groupby(
        'FECHA')['CASOS'].sum().reset_index()

    fig_tendencia_delitos = px.line(delitos_por_fecha, x='FECHA', y='CASOS', title='Tendencia de Delitos Violentos en el Tiempo en Chile', labels={
                                    'FECHA': 'Fecha', 'CASOS': 'Número de Casos'})
    fig_tendencia_delitos.update_layout(title={'xanchor': 'center', 'x': 0.5},)
    st.plotly_chart(fig_tendencia_delitos, use_container_width=True)

    # ------------------------------------------------
    # Tendencia de delitos a por tipo en el tiempo
    delitos_tipo_fecha = df.groupby(['FECHA', 'DELITO'])[
        'CASOS'].sum().reset_index()
    delitos_tipo_fecha.sort_values('FECHA', inplace=True)

    fig = px.line(delitos_tipo_fecha, x='FECHA', y='CASOS', color='DELITO',
                  title='Tendencia de Delitos Violentos por Tipo a lo Largo del Tiempo en Chile',
                  labels={'FECHA': 'Fecha', 'CASOS': 'Número de Casos', 'DELITO': 'Tipo de Delito'})
    fig.update_layout(xaxis_title='Fecha',
                      yaxis_title='Número de Casos',
                      legend_title='Tipo de Delito',
                      title={'xanchor': 'center', 'x': 0.5})
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------
    st.subheader("Análisis de detenciones")

    delitos_por_tipo = df.groupby(
        'DELITO')['CASOS'].sum().reset_index()
    delitos_por_tipo.sort_values('CASOS', ascending=False, inplace=True)

    fig = px.bar(delitos_por_tipo, x='DELITO', y='CASOS',
                 title='Distribución de Delitos por Tipo',
                 labels={'DELITO': 'Tipo de Delito', 'CASOS': 'Número de Casos'})
    fig.update_layout(xaxis_title='Tipo de Delito',
                      yaxis_title='Número de Casos',
                      xaxis_tickangle=-45,
                      title={'xanchor': 'center', 'x': 0.5}
                      )
    st.plotly_chart(fig, use_container_width=True)

    st.write("Distribución de delitos por tipo mostrando la frecuencia de cada tipo de delito en el conjunto de datos, lo que permite identificar los tipos de delitos más comunes.")
    st.divider()
    # ------------------------------------------------
    st.subheader("Distribución geográfica de delitos")

    delitos_por_region = df.groupby(
        'REGION')['CASOS'].sum().reset_index()
    delitos_por_region.sort_values('CASOS', ascending=False, inplace=True)

    fig = px.bar(delitos_por_region, x='REGION', y='CASOS',
                 title='Distribución Geográfica de Delitos por Región',
                 labels={'REGION': 'Región', 'CASOS': 'Número de Casos'})
    fig.update_layout(xaxis_title='Región',
                      yaxis_title='Número de Casos',
                      xaxis_tickangle=-45,
                      title={'xanchor': 'center', 'x': 0.5}
                      )
    st.plotly_chart(fig, use_container_width=True)

    st.write("Distribución de los delitos por regiones o comunas, resaltando las áreas con mayor incidencia de delitos.")
    st.divider()
    # ------------------------------------------------

    delitos_por_comuna = df.groupby(
        'COMUNA')['CASOS'].sum().reset_index()
    delitos_por_comuna.sort_values('CASOS', ascending=False, inplace=True)
    top_comunas = delitos_por_comuna.head(25)

    fig = px.bar(top_comunas, x='COMUNA', y='CASOS',
                 title='Top 25 Comunas con Mayor Incidencia de Delitos',
                 labels={'COMUNA': 'Comuna', 'CASOS': 'Número de Casos'},
                 color='CASOS',  # Usar el número de casos para el color de las barras
                 color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(xaxis_title='Comuna',
                      yaxis_title='Número de Casos',
                      xaxis_tickangle=-45,
                      title={'xanchor': 'center', 'x': 0.5}
                      )
    st.plotly_chart(fig, use_container_width=True)

    st.write("Distribución comunal de delitos: distribución de los delitos por 'comuna' resaltando las comunas con mayor incidencia de delitos.")
    st.divider()
    # ------------------------------------------------

    delitos_comuna_tipo = df.groupby(['COMUNA', 'DELITO'])[
        'CASOS'].sum().reset_index()
    top_comunas_list = delitos_por_comuna.head(10)['COMUNA'].tolist()
    delitos_comuna_tipo_top = delitos_comuna_tipo[delitos_comuna_tipo['COMUNA'].isin(
        top_comunas_list)]

    fig = px.bar(delitos_comuna_tipo_top, x='COMUNA', y='CASOS', color='DELITO',
                 title='Distribución de Delitos por Comuna, Diferenciando por Tipo de Delito',
                 labels={'COMUNA': 'Comuna', 'CASOS': 'Número de Casos', 'DELITO': 'Tipo de Delito'})
    fig.update_layout(xaxis_title='Comuna',
                      yaxis_title='Número de Casos',
                      legend_title='Tipo de Delito',
                      xaxis_tickangle=-45,
                      title={'xanchor': 'center', 'x': 0.5})
    st.plotly_chart(fig, use_container_width=True)

    st.write("Distribución comunal de delitos: distribución de los delitos por 'comuna' diferenciando los tipos de delitos, resaltando las comunas con mayor incidencia de delitos.")
    st.write("Este gráfico no solo destaca las comunas con la mayor incidencia total de delitos, sino que también muestra la proporción de diferentes tipos de delitos dentro de cada comuna, lo que proporciona una comprensión más profunda de la naturaleza de los delitos en estas áreas.")

with tabTable:
    # st.dataframe(df, height=500)
    st.write(df.head(10))

with tabIA:  # EDA IA
    st.subheader("Análisis exploratorio con Inteligencia Artificial")
    with st.expander("Información importante"):
        st.write("Las respuestas son generadas por un modelo de lenguaje de OpenAI, el cual permite realizar consultas sobre el dataset de MACEDA. Ingrese su consulta la que pudiera ser respondida por el modelo en forma de texto o una imagen gráfica.")
        st.write(
            "Por ejemplo, puede preguntar: ¿Cuántos eventos de tipo 'X' ocurrieron en la región 'Y' en el año '2018'?")
        st.warning(
            "*Nota*: Esta es una tecnología en experimentación por lo que las respuestas pueden no ser del todo exactas.")
    st.write("")
    # llm = OpenAI(api_token=API_KEY)
    llm = OpenAI(client=OpenAI, streaming=True,
                 api_token=API_KEY, temperature=0.5)

    # with cor:
    prompt = st.text_area("Ingrese su consulta:")

    if st.button("Generar respuesta"):
        if prompt:
            with st.spinner("Generando respuesta... por favor espere."):
                llm = OpenAI(api_token=os.environ["OPENAI_API_KEY"])
                # query = SmartDataframe(df, config={"llm": llm})
                query = Agent(df, config={"llm": llm,
                                          "save_charts": False,
                                          # "save_charts_path": user_path,
                                          "open-charts": True,
                                          "verbose": True,
                                          "response_parser": StreamlitResponse
                                          })

                response = query.chat(prompt)

                if isinstance(response, str) and response.endswith("png"):
                    st.image(response)
                else:
                    st.write(response)
        else:
            st.write("Por favor ingrese una consulta.")

pro_bar.empty()
# with tabBleau:
#     report = pgw.walk(df, return_html=True)
#     components.html(report, height=1000, scrolling=True)


with tabInfo:
    st.write("Información")

    st.write("Detenciones en Chile")
    st.write("Fecha de actualización: ",
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.write("Autor: Patricio Araneda")
    st.write("Fuente: [Fundación Chile 21](https://chile21.cl/)")
    st.write("San Sebastián 2807, Las Condes, Santiago de Chile")
