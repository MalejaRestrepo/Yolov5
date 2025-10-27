import streamlit as st
import torch
import numpy as np
import pandas as pd
import cv2

st.set_page_config(
    page_title="🔍 Detección de Objetos YOLOv5 (Lite)",
    page_icon="🧠",
    layout="wide"
)

st.title("🔍 Detección de Objetos con YOLOv5")
st.markdown("""
Esta app usa **YOLOv5** directamente desde Torch Hub, sin dependencias pesadas.  
Puedes subir o capturar una imagen y detectar objetos en tiempo real. 💜
""")

# ---------------- CARGAR MODELO ----------------
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

with st.spinner("Cargando modelo YOLOv5..."):
    model = load_model()

# ---------------- PARÁMETROS ----------------
conf = st.sidebar.slider("Nivel de confianza", 0.0, 1.0, 0.25, 0.01)
model.conf = conf

# ---------------- IMAGEN ----------------
option = st.radio("Selecciona una opción:", ["📤 Subir imagen", "📸 Usar cámara"])

if option == "📤 Subir imagen":
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        bytes_data = uploaded_file.read()
        np_img = np.frombuffer(bytes_data, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        results = model(image)
        results.render()
        st.image(image, channels="BGR", caption="Resultado YOLOv5", use_container_width=True)
        st.success("✅ Detección completada.")
        
        # Mostrar tabla
        df = results.pandas().xyxy[0]
        st.dataframe(df[["name", "confidence"]])
elif option == "📸 Usar cámara":
    picture = st.camera_input("Toma una foto")
    if picture:
        bytes_data = picture.getvalue()
        np_img = np.frombuffer(bytes_data, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        results = model(image)
        results.render()
        st.image(image, channels="BGR", caption="Resultado YOLOv5", use_container_width=True)
        st.success("✅ Detección completada.")
        
        df = results.pandas().xyxy[0]
        st.dataframe(df[["name", "confidence"]])

st.markdown("---")
st.caption("Desarrollado con 💜 usando Streamlit + YOLOv5 (Torch Hub)")

# ---------------- PIE DE PÁGINA ----------------
st.markdown("---")
st.caption("""
Desarrollado con 💜 usando Streamlit, YOLOv5 y PyTorch.
""")
