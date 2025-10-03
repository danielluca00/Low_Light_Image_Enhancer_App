import streamlit as st
from sqlalchemy import create_engine, text
import torch
from PIL import Image
from io import BytesIO
import json
import pandas as pd
import numpy as np

from models.cdan import CDAN
from models.model import Model

# --- Configurazione ---
with open("config/default.json", "r") as f: 
    config = json.load(f)

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

# --- Supabase/Postgres DB setup ---
DATABASE_URL = st.secrets["DATABASE_URL"]
engine = create_engine(DATABASE_URL)

def init_db():
    """Crea le tabelle users e feedback se non esistono."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                filename TEXT,
                preset TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

def add_user(username, password):
    try:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO users (username, password) VALUES (:u, :p)"),
                {"u": username, "p": password}
            )
    except Exception:
        st.warning("⚠️ Username già esistente o errore di connessione")

def login_user(username, password):
    with engine.begin() as conn:
        result = conn.execute(
            text("SELECT id FROM users WHERE username=:u AND password=:p"),
            {"u": username, "p": password}
        ).fetchone()
        return result[0] if result else None

def save_feedback(user_id, filename, preset):
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO feedback (user_id, filename, preset) VALUES (:uid, :fname, :preset)"),
            {"uid": user_id, "fname": filename, "preset": preset}
        )

# --- Funzione cosine similarity senza sklearn ---
def cosine_similarity_matrix(X):
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / (norm + 1e-8)
    return np.dot(X_normalized, X_normalized.T)

# --- Collaborative Filtering con fallback ---
def get_collaborative_recommendation(user_id):
    df = pd.read_sql("SELECT user_id, preset, COUNT(*) as freq FROM feedback GROUP BY user_id, preset", engine)
    
    if df.empty:
        return None

    matrix = df.pivot(index='user_id', columns='preset', values='freq').fillna(0)

    if user_id in matrix.index and len(matrix.index) > 1:
        sim_matrix = pd.DataFrame(cosine_similarity_matrix(matrix.values), index=matrix.index, columns=matrix.index)
        similar_users = sim_matrix[user_id].drop(user_id).sort_values(ascending=False).index

        weighted_scores = pd.Series(0, index=matrix.columns)
        for sim_user in similar_users:
            weighted_scores += sim_matrix.at[user_id, sim_user] * matrix.loc[sim_user]

        user_feedback = matrix.loc[user_id]
        weighted_scores = weighted_scores.where(user_feedback == 0, 0)

        if weighted_scores.sum() > 0:
            return weighted_scores.idxmax()

    if user_id in matrix.index:
        most_chosen = matrix.loc[user_id].idxmax()
        if matrix.loc[user_id].max() > 0:
            return most_chosen

    global_counts = df.groupby("preset")["freq"].sum()
    if not global_counts.empty:
        return global_counts.idxmax()

    return None

# --- Inizializza DB ---
init_db()

# --- Streamlit UI ---
st.set_page_config(page_title="Low-Light Enhancer", page_icon="✨", layout="wide")
st.title("🌙✨ Low-Light Image Enhancer")

# --- Stato utente ---
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "preset_choice" not in st.session_state:
    st.session_state["preset_choice"] = None
if "last_uploaded" not in st.session_state:
    st.session_state["last_uploaded"] = None

st.sidebar.subheader("🔐 Account")

if st.session_state["user_id"] is None:
    choice = st.sidebar.radio("Scegli un'opzione", ["Login", "Registrati"])
    username = st.sidebar.text_input("Username", key="username_input")
    password = st.sidebar.text_input("Password", type="password", key="password_input")

    if choice == "Login":
        if st.sidebar.button("Login"):
            user_id = login_user(username, password)
            if user_id:
                st.session_state["user_id"] = user_id
                st.session_state["username"] = username
                st.success(f"✅ Bentornato, {username}!")
                st.rerun()
            else:
                st.error("❌ Credenziali errate")
    else:
        if st.sidebar.button("Registrati"):
            add_user(username, password)
            user_id = login_user(username, password)
            if user_id:
                st.session_state["user_id"] = user_id
                st.session_state["username"] = username
                st.success("✅ Registrazione completata e login effettuato!")
                st.rerun()
else:
    st.sidebar.success(f"✅ Utente loggato: {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        st.session_state["user_id"] = None
        st.session_state["username"] = None
        st.rerun()

# --- App principale ---
if st.session_state["user_id"]:
    st.markdown("Carica un'immagine scura e confronta i miglioramenti proposti!")

    uploaded_file = st.file_uploader("📤 Carica un'immagine", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            img_check = Image.open(uploaded_file)
            img_check.verify()
            uploaded_file.seek(0)
            full_image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("⚠️ Il file caricato non è un'immagine valida. Usa PNG o JPG.")
            st.stop()

        original_size = full_image.size

        max_size = 800
        image_for_model = full_image.copy()
        if max(image_for_model.size) > max_size:
            image_for_model.thumbnail((max_size, max_size))

        with st.spinner("✨ Elaborazione con preset in corso..."):
            presets = {
                "Luminosità": model.infer(image_for_model, 1.0, 1.1, 0.2),
                "Contrasto": model.infer(image_for_model, 1.4, 1.0, 0.5),
                "Fedeltà cromatica": model.infer(image_for_model, 1.1, 1.4, 0.3),
                "Nitidezza": model.infer(image_for_model, 1.2, 1.1, 1.0)
            }

        recommended_preset = get_collaborative_recommendation(st.session_state["user_id"])
        options = ["-- Seleziona una versione --"] + list(presets.keys())

        if st.session_state.get("last_uploaded") != uploaded_file.name:
            st.session_state["preset_choice"] = recommended_preset or options[0]
            st.session_state["last_uploaded"] = uploaded_file.name

        if st.session_state.get("preset_choice") not in options:
            st.session_state["preset_choice"] = options[0]

        st.subheader("🌟 Confronta le varianti")
        cols = st.columns(len(presets))
        for (name, img), col in zip(presets.items(), cols):
            if name == recommended_preset:
                col.markdown(f"**✅ {name} (Consigliato)**")
            else:
                col.markdown(f"**{name}**")
            col.image(img.resize(original_size), use_container_width=True)

        preset_choice = st.radio(
            "Seleziona una versione:",
            options,
            key="preset_choice"
        )

        if preset_choice != "-- Seleziona una versione --":
            if st.button("📥 Salva preferenza e scarica immagine"):
                save_feedback(st.session_state["user_id"], uploaded_file.name, preset_choice)
                st.success(f"Feedback salvato! Hai scelto: **{preset_choice}** ✅")

                buf = BytesIO()
                presets[preset_choice].save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="📥 Scarica la versione scelta",
                    data=byte_im,
                    file_name=f"output_{preset_choice}.png",
                    mime="image/png"
                )
        else:
            st.info("Seleziona una versione prima di salvare.")

# --- Info ---
with st.expander("ℹ️ Come funziona"):
    st.write(""" 
        Questo modello utilizza una rete neurale **CDAN** per migliorare immagini 
        in condizioni di scarsa illuminazione.  
        Mostriamo 4 varianti (Luminosità, Contrasto, Fedeltà cromatica, Nitidezza) 
        e raccogliamo le tue preferenze collegate al tuo account. 
        Il preset consigliato viene selezionato automaticamente:
        - basato su utenti simili se disponibili
        - altrimenti sul preset più scelto dall’utente
        - oppure sul preset più popolare globalmente
    """)
