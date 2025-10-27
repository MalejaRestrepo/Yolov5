import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# ---------------- CONFIGURACIÓN DE PÁGINA ----------------
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

# ---------------- ESTILO VISUAL GLOBAL ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #e7deff 0%, #d4e8ff 100%);
    color: #1c1740;
    font-family: 'Poppins', sans-serif;
}
.block-container {
    background: #faf9ff;
    border: 1px solid #c8bfff;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    box-shadow: 0 10px 24px rgba(28, 23, 64, 0.12);
}
h1, h2, h3 {
    color: #2b1d59;
    text-align: center;
    font-weight: 700;
}
label, p, div, span {
    color: #1c1740 !important;
}
section[data-testid="stSidebar"] {
    background: #eee8ff;
    border-right: 2px solid #c8bfff;
}
section[data-testid="stSidebar"] * {
    color: #1e1c3a !important;
}
div.stButton > button {
    background: linear-gradient(90deg, #b7a2ff 0%, #9ee0ff 100%) !important;
    color: #1c1740 !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: 1px solid #a8b9ff !important;
    box-shadow: 0 6px 14px rgba(28, 23, 64, 0.18) !important;
    font-size: 16px !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #a28eff 0%, #89d4ff 100%) !important;
    transform: translateY(-1px);
}
textarea, .stTextInput input {
    background-color: #ffffff !important;
    color: #1c1740 !important;
    border-radius: 10px !important;
    border: 1px solid #bda5ff !important;
}
[data-testid="stHeader"] {
    background: linear-gradient(90deg, #846dff 0%, #a5d8ff 100%) !important;
    color: white !important;
    height: 3.5rem;
    box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.25);
}
table, .stDataFrame {
    border-radius: 10px !important;
    border: 1px solid #d1caff !important;
    background-color: white !important;
}
[data-testid="stSidebarNav"] {
    background: #f4eeff !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- FUNCIÓN PARA CARGAR EL MODELO ----------------
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            model = yolov5.load(model_path)
            return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        💡 **Recomendaciones**:
        1. Instalar una versión compatible de PyTorch y YOLOv5:
           ```
           pip install torch==1.12.0 torchvision==0.13.0
           pip install yolov5==7.0.9
           ```
        2. Asegúrate de tener el archivo `yolov5s.pt` en la ruta correcta.
        """)
        return None

# ---------------- TÍTULO ----------------
st.title("🔍 Detección de Objetos en Imágenes")
st.markdown("""
Esta aplicación utiliza **YOLOv5** para detectar objetos en imágenes capturadas con tu cámara.  
Ajusta los parámetros en la barra lateral para personalizar la detección. 💜
""")

# ---------------- CARGAR MODELO ----------------
with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

if model:
    # ---------------- BARRA LATERAL ----------------
    st.sidebar.title("⚙️ Parámetros de Detección")

    with st.sidebar:
        st.subheader("🎛️ Configuración")
        model.conf = st.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
        st.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")

        st.subheader("🧩 Opciones avanzadas")
        try:
            model.agnostic = st.checkbox('NMS class-agnostic', False)
            model.multi_label = st.checkbox('Múltiples etiquetas por caja', False)
            model.max_det = st.number_input('Detecciones máximas', 10, 2000, 1000, 10)
        except:
            st.warning("⚠️ Algunas opciones no están disponibles con esta versión.")

    # ---------------- CONTENEDOR PRINCIPAL ----------------
    main_container = st.container()

    with main_container:
        picture = st.camera_input("📸 Captura una imagen", key="camera")

        if picture:
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            with st.spinner("🔎 Detectando objetos..."):
                try:
                    results = model(cv2_img)
                except Exception as e:
                    st.error(f"Error durante la detección: {str(e)}")
                    st.stop()

            try:
                predictions = results.pred[0]
                boxes = predictions[:, :4]
                scores = predictions[:, 4]
                categories = predictions[:, 5]

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🖼️ Imagen con detecciones")
                    results.render()
                    st.image(cv2_img, channels='BGR', use_container_width=True)

                with col2:
                    st.subheader("📦 Objetos detectados")
                    label_names = model.names
                    category_count = {}

                    for category in categories:
                        idx = int(category.item()) if hasattr(category, 'item') else int(category)
                        category_count[idx] = category_count.get(idx, 0) + 1

                    data = []
                    for idx, count in category_count.items():
                        label = label_names[idx]
                        confidence = scores[categories == idx].mean().item() if len(scores) > 0 else 0
                        data.append({
                            "Categoría": label,
                            "Cantidad": count,
                            "Confianza promedio": f"{confidence:.2f}"
                        })

                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                        st.bar_chart(df.set_index('Categoría')['Cantidad'])
                    else:
                        st.info("No se detectaron objetos con los parámetros actuales.")
                        st.caption("Prueba bajando el umbral de confianza.")
            except Exception as e:
                st.error(f"Error al procesar los resultados: {str(e)}")
                st.stop()
else:
    st.error("❌ No se pudo cargar el modelo YOLOv5.")
    st.stop()

# ---------------- PIE DE PÁGINA ----------------
st.markdown("---")
st.caption("""
Desarrollado con 💜 usando Streamlit, YOLOv5 y PyTorch.
""")
