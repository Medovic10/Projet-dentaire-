import streamlit as st
import pandas as pd
import os
from PIL import Image

# Configuration des chemins
CSV_PATH = r"D:\te\prj_dentaire\patients_diagnostics.csv"
DATASET_PATH = r"D:\te\PRJ\PRJ"

# Liste élargie des classes (avec tes nouveaux diagnostics)
CLASSES = [
    "Normal", 
    "Carie", 
    "Fracture", 
    "Lesion_periapicale", 
    "Perte_osseuse", 
    "Pulpite_aigue", 
    "Maladie_parodontale"
]

st.set_page_config(page_title="Dental Labeler", layout="wide")

st.title("🦷 Assistant de Labellisation Clinique (Multi-Diagnostics)")
st.write("Cet outil vous permet de parcourir les patients, de visualiser leurs radiographies et d'enregistrer un ou plusieurs diagnostics simultanément.")

# Chargement du CSV
if not os.path.exists(CSV_PATH):
    st.error(f"Le fichier {CSV_PATH} n'existe pas. Veuillez d'abord exécuter 'prepare_sheet.py'.")
    st.stop()

# Utilisation du session_state pour stocker le DataFrame
if 'df' not in st.session_state:
    st.session_state.df = pd.read_csv(CSV_PATH, sep=';', encoding='utf-8-sig', keep_default_na=False)

df = st.session_state.df

# Liste des patients
patient_list = df['Dossier_Patient'].tolist()

# Index du patient actuel
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

# Synchroniser l'index lors d'un changement dans la barre latérale
def on_sidebar_change():
    st.session_state.current_index = patient_list.index(st.session_state.sidebar_patient)

selected_patient = st.sidebar.selectbox(
    "🔍 Sélectionner un patient", 
    patient_list, 
    index=st.session_state.current_index,
    key="sidebar_patient",
    on_change=on_sidebar_change
)

current_idx = st.session_state.current_index
patient_row = df.iloc[current_idx]
patient_name = patient_row['Dossier_Patient']

st.header(f"👤 Patient Actuel : {patient_name}")

# Layout : Gauche = Formulaire, Droite = Radiographies
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📝 Diagnostics & Notes")
    
    # Récupérer la chaîne des diagnostics déjà enregistrés
    current_diag_str = str(patient_row['Diagnostic_Principal']).strip()
    
    # Convertir la chaîne en liste pour pré-remplir le composant multiselect
    if current_diag_str == "" or pd.isna(patient_row['Diagnostic_Principal']):
        default_vals = []
    else:
        default_vals = [d.strip() for d in current_diag_str.split(",") if d.strip() in CLASSES]
    
    # Composant de sélection multiple (Multiselect)
    selected_diags = st.multiselect(
        "Diagnostics détectés",
        options=CLASSES,
        default=default_vals,
        help="Sélectionnez toutes les pathologies applicables à ce patient."
    )

    current_notes = patient_row['Notes_Cliniques']
    notes = st.text_area("Notes Cliniques / Observations", value=current_notes, height=150)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        save_clicked = st.button("💾 Sauvegarder", type="primary", width="stretch")
    with col_btn2:
        save_next_clicked = st.button("💾 Sauvegarder & Suivant", type="secondary", width="stretch")

    if save_clicked or save_next_clicked:
        # Joindre les choix par une virgule pour les stocker proprement dans une seule cellule CSV
        val_to_save = ", ".join(selected_diags)
        
        # Mise à jour du DataFrame en mémoire
        st.session_state.df.at[current_idx, 'Diagnostic_Principal'] = val_to_save
        st.session_state.df.at[current_idx, 'Notes_Cliniques'] = notes
        
        # Sauvegarde physique sur le disque
        st.session_state.df.to_csv(CSV_PATH, sep=';', index=False, encoding='utf-8-sig')
        st.toast(f"Données enregistrées pour {patient_name} !", icon="💾")
        
        if save_next_clicked and current_idx < len(patient_list) - 1:
            st.session_state.current_index += 1
        st.rerun()

    # Boutons de navigation
    st.write("---")
    nav_cols = st.columns(2)
    with nav_cols[0]:
        if st.button("⬅️ Précédent", disabled=(current_idx == 0), width="stretch"):
            st.session_state.current_index -= 1
            st.rerun()
    with nav_cols[1]:
        if st.button("Suivant ➡️", disabled=(current_idx == len(patient_list) - 1), width="stretch"):
            st.session_state.current_index += 1
            st.rerun()

    # Barre de progression globale
    st.write("---")
    completed = df[df['Diagnostic_Principal'].notna() & (df['Diagnostic_Principal'] != "")].shape[0]
    total = len(df)
    st.metric("Progression", f"{completed} / {total} Patients labellisés", f"{completed/total*100:.1f}%")
    st.progress(completed / total)

with col2:
    st.subheader(f"🖼️ Radiographies ({patient_row['Nombre_Radios']} images)")
    
    patient_folder = os.path.join(DATASET_PATH, patient_name)
    if os.path.exists(patient_folder):
        images = [f for f in os.listdir(patient_folder) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]
        if images:
            img_cols = st.columns(2)
            for i, img_name in enumerate(images):
                img_path = os.path.join(patient_folder, img_name)
                col_to_use = img_cols[i % 2]
                try:
                    img = Image.open(img_path)
                    col_to_use.image(img, caption=img_name, width="stretch")
                except Exception as e:
                    col_to_use.error(f"Erreur d'ouverture de l'image {img_name}")
        else:
            st.warning("Aucune image trouvée dans ce dossier.")
    else:
        st.error(f"Le dossier du patient est introuvable au chemin : {patient_folder}")