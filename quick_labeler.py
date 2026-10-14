import os
import pandas as pd
import streamlit as st
from PIL import Image
import sys
import asyncio

# Utiliser le gestionnaire d'événements Selector sous Windows pour éviter WinError 10054
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

CSV_PATH = "data/dataset_metadata.csv"
CLASSES = ["Carie", "Fracture", "Lesion_periapicale", "Perte_osseuse", "Pulpite_aigue", "Maladie_parodontale"]

st.set_page_config(page_title="Quick Dental Labeler", layout="wide")

if not os.path.exists(CSV_PATH):
    st.error("Fichier dataset_metadata.csv introuvable.")
    st.stop()

df = pd.read_csv(CSV_PATH)

# Trouver les images non encore annotées (toutes les classes sont à 0)
unlabeled_mask = df[CLASSES].sum(axis=1) == 0
unlabeled_indices = df[unlabeled_mask].index.tolist()

st.title("⚡ Annotation Rapide des Radiographies")
st.write(f"**Progrès :** {len(df) - len(unlabeled_indices)} / {len(df)} images annotées ({len(unlabeled_indices)} restantes)")

if not unlabeled_indices:
    st.success("🎉 Toutes les images sont annotées !")
    st.stop()

# Gestion de l'index de l'image courante
if "current_idx_pos" not in st.session_state:
    st.session_state.current_idx_pos = 0

if st.session_state.current_idx_pos >= len(unlabeled_indices):
    st.session_state.current_idx_pos = 0

current_df_idx = unlabeled_indices[st.session_state.current_idx_pos]
row = df.loc[current_df_idx]
img_path = row['image_path']

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Image [{st.session_state.current_idx_pos + 1}/{len(unlabeled_indices)}] : {img_path}")
    if os.path.exists(img_path):
        st.image(Image.open(img_path), use_container_width=True)
    else:
        st.error(f"Fichier image introuvable : {img_path}")

with col2:
    st.subheader("Sélectionnez la/les pathologie(s) :")
    
    selected_classes = {}
    for cls in CLASSES:
        selected_classes[cls] = st.checkbox(cls, value=bool(row[cls]))

    if st.button("💾 Enregistrer & Image Suivante ➔", type="primary", use_container_width=True):
        for cls, val in selected_classes.items():
            df.at[current_df_idx, cls] = 1 if val else 0
        
        df.to_csv(CSV_PATH, index=False)
        st.session_state.current_idx_pos += 1
        st.rerun()

    if st.button("⏭️ Passer (Ignorer)", use_container_width=True):
        st.session_state.current_idx_pos += 1
        st.rerun()