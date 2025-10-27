import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
import pandas as pd

st.set_page_config(page_title="🔍 Detección de Objetos YOLOv8", page_icon="🧠", layout="wide")

st.title("🔍 Detección de Objetos con YOLOv8 (Ultralytics)")
st.markdown("""
Esta app usa **YOLOv8** para detectar objetos en imágenes o desde la cámara.  
Es ligera, compatible con Streamlit Cloud y no requiere GPU. 💜
""")

# Cargar modelo YOLOv8 preentrenado
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # modelo más liviano

with st.spinner("Cargando modelo YOLOv8..."):
    model = load_model()

# Sidebar con parámetros
conf = st.sidebar.slider("Nivel de confianza", 0.0, 1.0, 0.25, 0.01)
st.sidebar.caption(f"Confianza actual: {conf:.2f}")

# Selector de modo
modo = st.radio("Selecciona una opción:", ["📤 Subir imagen", "📸 Usar cámara"])

if modo == "📤 Subir imagen":
    file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if file:
        bytes_data = file.read()
        np_img = np.frombuffer(bytes_data, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        results = model.predict(image, conf=conf)
        annotated = results[0].plot()
        st.image(annotated, channels="BGR", caption="Resultado YOLOv8", use_container_width=True)

        # Mostrar detecciones
        boxes = results[0].boxes.data.cpu().numpy()
        if len(boxes) > 0:
            df = pd.DataFrame(boxes, columns=["x1", "y1", "x2", "y2", "confianza", "clase"])
            df["nombre"] = [model.names[int(c)] for c in df["clase"]]
            st.dataframe(df[["nombre", "confianza"]].round(2))
        else:
            st.info("No se detectaron objetos.")

elif modo == "📸 Usar cámara":
    foto = st.camera_input("Toma una foto con tu cámara")
    if foto:
        bytes_data = foto.getvalue()
        np_img = np.frombuffer(bytes_data, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        results = model.predict(image, conf=conf)
        annotated = results[0].plot()
        st.image(annotated, channels="BGR", caption="Resultado YOLOv8", use_container_width=True)

        boxes = results[0].boxes.data.cpu().numpy()
        if len(boxes) > 0:
            df = pd.DataFrame(boxes, columns=["x1", "y1", "x2", "y2", "confianza", "clase"])
            df["nombre"] = [model.names[int(c)] for c in df["clase"]]
            st.dataframe(df[["nombre", "confianza"]].round(2))
        else:
            st.info("No se detectaron objetos.")

st.markdown("---")
st.caption("💜 Desarrollado con Streamlit + Ultralytics YOLOv8")
