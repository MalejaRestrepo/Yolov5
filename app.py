import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
import pandas as pd

# CONFIGURACIÓN GENERAL
st.set_page_config(
    page_title="🔍 Detección de Objetos YOLOv8",
    page_icon="🧠",
    layout="wide"
)

# ESTILOS LAVANDA-VIOLETA (MISMA LÍNEA VISUAL)
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #e8dcff 0%, #d7c4ff 100%);
        color: #22143d;
        font-family: 'Poppins', sans-serif;
    }

    .block-container {
        background: #faf7ff;
        border: 1px solid #cbb3ff;
        border-radius: 16px;
        padding: 2rem 2.2rem;
        box-shadow: 0 10px 24px rgba(34, 20, 61, 0.12);
    }

    h1, h2, h3 {
        color: #3b2168;
        text-align: center;
        font-weight: 700;
    }

    p, li, label {
        color: #22143d;
    }

    section[data-testid="stSidebar"] {
        background: #efe6ff;
        border-right: 2px solid #c9b1ff;
        color: #2a1d5c;
    }

    section[data-testid="stSidebar"] * {
        color: #2a1d5c !important;
        font-size: 15px;
    }

    div.stButton > button {
        background-color: #8b6aff;
        color: white !important;
        font-weight: 700;
        border-radius: 10px;
        border: 1px solid #6f51ea;
        box-shadow: 0 6px 14px rgba(34, 20, 61, 0.18);
        font-size: 16px;
        padding: 9px 24px;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background-color: #6f51ea;
        transform: translateY(-1px);
    }

    div[data-baseweb="select"] {
        background-color: #ffffff !important;
        color: #22143d !important;
        border-radius: 10px !important;
        border: 1px solid #bda5ff !important;
    }

    div[data-baseweb="select"] * {
        color: #22143d !important;
    }

    audio, img {
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }

    [data-testid="stHeader"] {
        background: linear-gradient(90deg, #5a3ccf 0%, #7b59e3 100%) !important;
        color: white !important;
        height: 3.5rem;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.25);
    }

    [data-testid="stToolbar"] {
        right: 1rem;
        top: 0.5rem;
        color: white !important;
    }

    .css-1q8dd3e { color: #ffffff !important; } /* texto sobre fondos violetas */
    </style>
""", unsafe_allow_html=True)

# TÍTULO PRINCIPAL
st.title("🔍 Detección de Objetos con YOLOv8 (Ultralytics)")
st.markdown("""
Esta app utiliza **YOLOv8** para detectar objetos en imágenes o desde la cámara.  
Es ligera, compatible con Streamlit Cloud y visualmente coherente con la línea lavanda 💜.
""")

# CARGAR MODELO
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

with st.spinner("Cargando modelo YOLOv8..."):
    model = load_model()

# CONFIGURACIONES EN SIDEBAR
st.sidebar.header("🎚️ Parámetros de Detección")
conf = st.sidebar.slider("Nivel de confianza", 0.0, 1.0, 0.25, 0.01)
st.sidebar.caption(f"Confianza actual: {conf:.2f}")

# OPCIONES DE INTERACCIÓN
modo = st.radio("Selecciona una opción:", ["📤 Subir imagen", "📸 Usar cámara"])

if modo == "📤 Subir imagen":
    file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if file:
        bytes_data = file.read()
        np_img = np.frombuffer(bytes_data, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        with st.spinner("Detectando objetos..."):
            results = model.predict(image, conf=conf)
            annotated = results[0].plot()

        st.image(annotated, channels="BGR", caption="Resultado YOLOv8", use_container_width=True)

        boxes = results[0].boxes.data.cpu().numpy()
        if len(boxes) > 0:
            df = pd.DataFrame(boxes, columns=["x1", "y1", "x2", "y2", "confianza", "clase"])
            df["nombre"] = [model.names[int(c)] for c in df["clase"]]
            st.markdown("### 📋 Objetos detectados")
            st.dataframe(df[["nombre", "confianza"]].round(2))
        else:
            st.info("No se detectaron objetos.")

elif modo == "📸 Usar cámara":
    foto = st.camera_input("Toma una foto con tu cámara")
    if foto:
        bytes_data = foto.getvalue()
        np_img = np.frombuffer(bytes_data, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        with st.spinner("Detectando objetos..."):
            results = model.predict(image, conf=conf)
            annotated = results[0].plot()

        st.image(annotated, channels="BGR", caption="Resultado YOLOv8", use_container_width=True)

        boxes = results[0].boxes.data.cpu().numpy()
        if len(boxes) > 0:
            df = pd.DataFrame(boxes, columns=["x1", "y1", "x2", "y2", "confianza", "clase"])
            df["nombre"] = [model.names[int(c)] for c in df["clase"]]
            st.markdown("### 📋 Objetos detectados")
            st.dataframe(df[["nombre", "confianza"]].round(2))
        else:
            st.info("No se detectaron objetos.")

st.markdown("---")
st.caption("💜 Desarrollado con Streamlit + Ultralytics YOLOv8")
