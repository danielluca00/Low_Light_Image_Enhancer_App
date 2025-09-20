import streamlit as st 
import json 
import torch
from PIL import Image
from io import BytesIO

from models.cdan import CDAN
from models.model import Model

# --- Configurazione --- 
with open("config/default.json", "r") as f: 
    config = json.load(f)

# Forziamo l'uso della fase 'test' e della CPU
config['phase'] = 'test'
device = torch.device('cpu') 
config[config['phase']]['device'] = 'cpu'

# --- Caricamento modello ---
network = CDAN().to(device) 
model = Model(network=network, config=config, dataloader=None) 
model.network.load_state_dict(torch.load(
    f"{config[config['phase']]['model_path']}/{config[config['phase']]['model_name']}", 
    map_location=device
))

# --- Streamlit UI ---
st.set_page_config(page_title="Low-Light Enhancer", page_icon="✨", layout="wide")
st.title("🌙✨ Low-Light Image Enhancer")
st.markdown("Carica un'immagine scura e migliorala automaticamente!")

uploaded_file = st.file_uploader("📤 Carica un'immagine", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    original_size = image.size  # Salviamo dimensioni originali

    # 🔹 Ridimensionamento sicuro se l'immagine è troppo grande
    max_size = 800
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))
    
    # Parametri personalizzabili
    st.sidebar.header("⚙️ Parametri di miglioramento") 
    contrast_factor = st.sidebar.slider("Contrasto", 0.5, 2.0, 1.03, 0.01)
    saturation_factor = st.sidebar.slider("Saturazione", 0.5, 2.0, 1.55, 0.01) 
    sharpen_strength = st.sidebar.slider("Sharpening", 0.0, 3.0, 0.0, 0.01)

    # Elaborazione con spinner
    with st.spinner("✨ Miglioramento in corso..."):
        output_image = model.infer(
            image,
            contrast_factor=contrast_factor,
            saturation_factor=saturation_factor,
            sharpen_strength=sharpen_strength if sharpen_strength > 0 else None
        )

    st.success("✅ Elaborazione completata!")

    # Layout a due colonne per confronto
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Immagine originale")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🌟 Immagine migliorata")
        st.image(output_image, use_container_width=True)

    # Pulsante di download 
    buf = BytesIO()
    output_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label="📥 Scarica immagine migliorata",
        data=byte_im,
        file_name="output.png",
        mime="image/png"
    )

# --- Sezioni informative --- 
with st.expander("ℹ️ Come funziona"):
    st.write(""" Questo modello utilizza una rete neurale basata su 
             **CDAN** per migliorare immagini in condizioni di 
             scarsa illuminazione. Durante il processo, 
             vengono applicate correzioni di contrasto e colore 
             per rendere l'immagine più nitida e luminosa. 
             """)
