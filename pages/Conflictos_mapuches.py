'''
Conflictos mapuches en Chile basadod en datos de MACEDA (Mapuche Chilean State Conflict Event Database).
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

# import base64
# from io import BytesIO
# from PIL import Image

import plotly.express as px
import plotly.graph_objects as go

from millify import millify
import pygwalker as pgw
# from pandasai import SmartDatalake  # para multiples dataframes
from pandasai import Agent
from pandasai.llm.openai import OpenAI
from pandasai.responses.streamlit_response import StreamlitResponse

from datetime import datetime
from funciones import menu_pages, menu_inicio, load_data_csv
from dotenv import load_dotenv


# configuration
st.set_page_config(
    page_title="Maceda",
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


def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)

    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))


menu_pages()
# carga de datos

df = load_data_csv("https://data.vgclab.cl/public_data/dataset_maceda.csv")

# df = pd.read_csv(datos, sep=",", noindex=True, low_memory=False)
# sacar columnas
df = df.drop(columns=["id_evento", "fecha_reportada"])
df = df.rename(columns={'ubicacion_tipo': 'ubicacion',
               'evento_tipo_maceda': 'tipo de evento',
                        'evento_especifico': 'detalle de evento',
                        'actor_tipo_1': 'tipo de actor',
                        'actor_especifico_1': 'actor principal',
                        'actor_especifico_1_armas': 'armas',
                        'actor_tipo_2': 'actor afectado',
                        'actor_mapuche': 'actor mapuche',
                        'mapuche_identificado': 'mapuche identificado',
                        'mesn': 'mes'})

df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
# cambiar fecha al inicio
fecha = df.pop("fecha")
df.insert(0, "fecha", fecha)

# crear dataframe estatico, independiente de selecciones del usuario
serie = df.groupby(['fecha', 'tipo de evento']
                   ).size().reset_index(name='cuenta')
# medios = df.sort_values(by="medio").medio.unique()
# medios.insert(0, "Todos")

# sidebar
years = df["fecha"].dt.year.unique().tolist()
years.insert(0, "Todos")
anual = st.sidebar.selectbox("Seleccione un año", years)

if anual != 'Todos':
    df = df[df["fecha"].dt.year == anual]
else:
    pass

tipos_eventos = df["tipo de evento"].unique().tolist()
tipo_evento = st.sidebar.multiselect(
    "Seleccione un tipo de evento", tipos_eventos, default=tipos_eventos)

if tipo_evento != None:
    df = df[df["tipo de evento"].isin(tipo_evento)]
else:
    df = df


regiones = df["region"].unique().tolist()
regiones = [x for x in regiones if str(x) != 'nan']
region = st.sidebar.multiselect(
    "Seleccione una o mas regiones", regiones, default=regiones)

if region != None:
    df = df[df["region"].isin(region)]
else:
    df = df

# --------------------------
# METRICAS
# --------------------------
eventos_totales = df["tipo de evento"].value_counts().tolist()
eventos_totales = sum(eventos_totales)

regiones_totales = df["region"].unique().tolist()
regiones_totales = len(regiones_totales)

actores_totales = df["actor principal"].unique().tolist()
actores_totales = len(actores_totales)

ataques = df[df["tipo de evento"] == "ATAQUE"]
ataques = ataques["tipo de evento"].value_counts().tolist()
ataques = sum(ataques)

protesta_violenta = df[df["detalle de evento"] == "PROTESTA VIOLENTA"]
protesta_violenta = protesta_violenta["tipo de evento"].value_counts().tolist()
protesta_violenta = sum(protesta_violenta)


# grafica tipos de eventos (tipo)
eventos = df["tipo de evento"].unique().tolist()
eventos_pie = go.Figure(data=[go.Pie(
    labels=eventos, values=df["tipo de evento"].value_counts().tolist(), hole=.3)])

# actores involucrados
df_actor1 = df.groupby('actor principal').size().reset_index(name='cuenta')
df_actor1.plot(kind='barh', x='actor principal', y='cuenta',
               title='actor principal', figsize=(15, 8))

# grafica regiones (linea)
regiones = df["region"].unique().tolist()
regiones_pie = go.Figure(data=[go.Pie(
    labels=regiones, values=df["region"].value_counts().tolist(), hole=.3)])

# grafica eventos
tipos = df["tipo de evento"].unique().tolist()
tipos_bar = go.Figure(data=[go.Pie(
    labels=tipos, values=df["tipo de evento"].value_counts().tolist(), hole=.3)])

eventos_tiempo = px.line(
    df, x=df["fecha"], y=df["tipo de evento"], color=df["tipo de evento"])

# --------------------------
# MAIN
# --------------------------
st.subheader("Conflictos Mapuches en Chile - MACEDA")

tabPanel, tabTable, tabIA, tabInfo = st.tabs(
    ["Panel", "Tabla", "IA-EDA", "Información"])

with tabPanel:  # graficos

    col1, col2, col3, col4, col5 = st.columns(5, gap="medium")
    col1.metric("Eventos", millify(eventos_totales, precision=2))
    col2.metric("Regiones", regiones_totales)
    col3.metric("Tipo de Actores", actores_totales)
    col4.metric(label="Ataques", value=ataques)
    col5.metric(label="Protestas violentas", value=protesta_violenta)

    style_metric_cards()

    col1, col2 = st.columns(2, gap="medium")
    with col1:

        fig = px.pie(df, values=df["tipo de evento"].value_counts().tolist(), names=df["tipo de evento"].unique().tolist(),
                     title='Tipo de eventos')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # st.subheader("Regiones")
        fig = px.pie(df, values=df["region"].value_counts().tolist(), names=df["region"].unique().tolist(),
                     title='Regiones involucradas')
        st.plotly_chart(fig, use_container_width=True)

    # Actores involucrados

    df_actor1 = df.groupby('actor principal').size().reset_index(name='cuenta')

    fig = px.bar(
        df_actor1,
        x="actor principal",
        y="cuenta",
        color="actor principal",
        title="Actores participantes",
    )
    fig.update_layout(showlegend=False, xaxis_title='Tipo de actor')
    st.plotly_chart(fig, use_container_width=True)

    # cantidad de conflictos por tipo de evento

    # serial = serie.groupby(['tipo de evento']).size().reset_index(name='cuenta')
    serial = df.groupby('tipo de evento').size().reset_index(name='cuenta')
    fig = px.bar(
        serial,
        x="tipo de evento",
        y="cuenta",
        title="Cantidad de conflictos",
        color="tipo de evento"
    )
    fig.update_layout(showlegend=False, xaxis_title='Tipo de evento')
    st.plotly_chart(fig, use_container_width=True)

    # eventos en el tiempo

    # dt = df.groupby([(df.fecha.dt.year), (df.fecha.dt.month)])['tipo'].count()
    # dt.plot(kind='line', figsize=(15, 5), title='Cantidad de eventos por año')

    serial = df.groupby(['fecha', 'tipo de evento']
                        ).size().reset_index(name='cuenta')
    serial['year'] = pd.DatetimeIndex(serial['fecha']).year
    serial['month'] = serial['fecha'].dt.strftime('%m')
    serial['mes'] = serial['fecha'].dt.strftime('%Y-%m')

    serial = serial.groupby(['mes', 'tipo de evento'],
                            as_index=False)['cuenta'].sum()

    figserial = px.line(serial, x="mes", y="cuenta", line_group="tipo de evento",
                        color="tipo de evento", markers=True)
    figserial.update_layout(title='Desarrrollo de eventos en el tiempo',
                            xaxis_title='Fecha',
                            yaxis_title='No. de eventos',
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                title="",
                                yanchor="bottom",
                                y=1,
                                xanchor="right",
                                x=1,
                                traceorder="reversed",
                                title_font_family="Times New Roman",
                                font=dict(
                                    family="Courier",
                                    size=11,
                                    color="black"
                                ),
                                # bgcolor="LightSteelBlue",
                                # bordercolor="Black",
                                borderwidth=1
                            ))

    st.plotly_chart(figserial, use_container_width=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        # REGIONES

        df_actor1 = df.groupby('region').size().reset_index(name='cuenta')

        fig = px.bar(
            df_actor1,
            x="region",
            y="cuenta",
            color="region",
            title="Casos por regiones",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        st.write("Como es esperable, los eventos son más frecuentes en el conflicto mapuche-estado chileno, dentro de la región de la Araucanía, donde se concentra la mayor cantidad de eventos. Donde el 99,5% de la población indigena pertenece a la etnia mapuche, representando el 23,4% de la población total de esta región. ")
        st.markdown("Fuente: [https://oes.ufro.cl/index.php/oes-ufro/estudios-regionales](https://oes.ufro.cl/index.php/oes-ufro/estudios-regionales/estudios-externos/category/12-estudiosexternos#:~:text=En%20la%20regi%C3%B3n%20de%20La%20Araucan%C3%ADa%2C%20del%20total%20de%20ind%C3%ADgenas,poblaci%C3%B3n%20total%20de%20esta%20regi%C3%B3n).")

    with col2:
        # ACTOR AFECTADO

        df_afectado = df.groupby(
            'actor afectado').size().reset_index(name='cuenta')

        fig = px.bar(
            df_afectado,
            x="actor afectado",
            y="cuenta",
            color="actor afectado",
            title="Actores afectados",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        st.write("En cuanto a los actores afectados, se observa que la mayor cantidad de eventos afectan a la población civil, seguido por la policía y los actores no estatales. ")

    # COMUNA
    st.write("")
    st.subheader("Casos por comuna")
    with st.expander("Umbral de casos"):
        casos = st.slider(
            "Seleccione cantidad de casos mínimos a considerar", 10, 1000, 50)

    df = df[df.groupby('comuna').comuna.transform('size') > casos]
    df_comuna = df.groupby('comuna').size().reset_index(name='casos')
    fig = px.bar(
        df_comuna,
        x="comuna",
        y="casos",
        color="comuna",
        title=None,
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.write("La comuna de Ercilla, en la región de la Araucanía, es la que presenta la mayor cantidad de eventos, seguida por la comuna de Collipulli. ")

# --------------------------
with tabTable:  # tabla de datos
    # --------------------------
    st.dataframe(df, height=500)

# --------------------------
with tabIA:  # EDA IA
    # --------------------------
    st.subheader("Análisis exploratorio con Inteligencia Artificial")
    with st.expander("Información importante"):
        st.write("Las respuestas son generadas por un modelo de lenguaje de OpenAI, el cual permite realizar consultas sobre el dataset de MACEDA. Ingrese su consulta la que pudiera ser respondida por el modelo en forma de texto o una imagen gráfica.")
        st.write(
            "Por ejemplo, puede preguntar: ¿Cuántos eventos de tipo 'X' ocurrieron en la región 'Y' en el año '2018'?")
        st.info(
            "*Nota*: Esta es una tecnología en experimentación por lo que las respuestas pueden no ser del todo exactas.")
    st.write("")
    user_path = os.getcwd()
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

# --------------------------
# with tabBleau:  # graficos personalizados
#     report = pgw.walk(df, return_html=True)
#     components.html(report, height=1000, scrolling=True)
# --------------------------

with tabInfo:
    col, cor = st.columns(2, gap="medium")
    with col:
        st.write("El Proyecto de Datos Mapuche (MDP) tiene como objetivos identificar, digitalizar, compilar, procesar y armonizar información cuantitativa respecto al pueblo Mapuche. ")

        st.write("Basados en MACEDA (Mapuche Chilean State Conflict Event Database), primer registro sistemático de eventos relacionados al conflicto entre el pueblo mapuche y el estado chileno.\n")

        st.write("MPD reporta información del conflicto entre el Estado Chileno y el pueblo mapuche. La Base de Datos de Eventos sobre el Conflicto Mapuche-Estado Chileno MACEDA (por su acrónimo en inglés) reporta más de 4500 eventos para el período 1990-2021.")
        st.write("")
        st.markdown(
            "[Descargar datos (csv 2.2MB)](https://sites.google.com/view/danyjaimovich/links/mdp)")
        st.markdown(
            "[Documentación (377KB)](https://data.vgclab.cl/public_data/mdp_conflicto_maceda_codigos.pdf)")
        st.write("")
        st.write("Cómo citar:")
        st.write("Cayul, P., A. Corvalan, D. Jaimovich, and M. Pazzona (2022). Introducing MACEDA: New Micro-Data on an Indigenous Self-Determination Conflict. Journal of Peace Research ")

    with cor:
        st.image("images/maceda.jpg", width=500)
