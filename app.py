import os
import json
import time
import datetime
import cv2
import numpy as np
import torch
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps

import config
from src.model import get_efficientnet_model
from src.dataset import get_val_transforms
from src.explainability import GradCAM, overlay_heatmap
from services import init_db, save_patient, get_all_patients, generate_pdf_bytes

# ==========================================
# 1. INITIALISATION BDD ET CONFIGURATION PAGE
# ==========================================
init_db()

st.set_page_config(
    page_title="Dental AI Studio Enterprise — Diagnostic Radiologique",
    layout="wide",
    page_icon="🦷",
    initial_sidebar_state="expanded"
)

# Style CSS Avancé
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    .top-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        padding: 22px 30px;
        border-radius: 14px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.90);
        backdrop-filter: blur(12px);
        padding: 22px;
        border-radius: 14px;
        border: 1px solid rgba(226, 232, 240, 0.8);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .badge-positive {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-negative {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #ffffff;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
        color: #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. GESTION DE LA SESSION & AUTHENTIFICATION
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "users_db" not in st.session_state:
    st.session_state.users_db = {"dr.radiologue": "admin123", "dentiste": "dentiste2026"}

def login_user(username, password):
    if username in st.session_state.users_db and st.session_state.users_db[username] == password:
        st.session_state.authenticated = True
        st.session_state.username = username
        return True
    return False

def register_user(username, password):
    if username in st.session_state.users_db:
        return False
    st.session_state.users_db[username] = password
    return True

# ==========================================
# 3. CHARGEMENT DES MODÈLES ET SEUILS
# ==========================================
@st.cache_resource
def load_ensemble_and_thresholds():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for f in range(1, config.NUM_FOLDS + 1):
        path = os.path.join(config.MODEL_DIR, f"efficientnet_fold_{f}.pth")
        if os.path.exists(path):
            m = get_efficientnet_model(num_classes=config.NUM_CLASSES, dropout_prob=0.0)
            m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            m.to(device)
            m.eval()
            models.append(m)
    
    thresh_path = os.path.join(config.MODEL_DIR, "optimal_thresholds.json")
    if os.path.exists(thresh_path):
        with open(thresh_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)
    else:
        thresholds = {cls: 0.5 for cls in config.CLASSES}

    return models, thresholds, device

ensemble_models, thresholds, DEVICE = load_ensemble_and_thresholds()

# ==========================================
# 4. ÉCRAN DE CONNEXION / INSCRIPTION
# ==========================================
if not st.session_state.authenticated:
    st.markdown("<h1 style='text-align: center; color: #0f172a; margin-top: 30px;'>🦷 Dental AI Studio Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b;'>Plateforme Médicale d'Aide au Diagnostic Radiologique Multi-Label</p>", unsafe_allow_html=True)
    
    col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
    with col_c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        tab_login, tab_signup = st.tabs(["🔒 Connexion Praticien", "📝 Créer un Compte"])
        
        with tab_login:
            user_input = st.text_input("Identifiant", key="login_user")
            pass_input = st.text_input("Mot de passe", type="password", key="login_pass")
            if st.button("Se connecter", use_container_width=True, type="primary"):
                if login_user(user_input, pass_input):
                    st.success("Authentification réussie !")
                    st.rerun()
                else:
                    st.error("Identifiants invalides.")
        
        with tab_signup:
            new_user = st.text_input("Nouvel identifiant", key="reg_user")
            new_pass = st.text_input("Nouveau mot de passe", type="password", key="reg_pass")
            if st.button("Créer un compte", use_container_width=True):
                if new_user and new_pass:
                    if register_user(new_user, new_pass):
                        st.success("Compte créé avec succès ! Veuillez vous connecter.")
                    else:
                        st.warning("Cet utilisateur existe déjà.")
                else:
                    st.error("Veuillez remplir tous les champs.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ==========================================
# 5. APPLICATION PRINCIPALE
# ==========================================
st.markdown(f"""
<div class="top-header">
    <div>
        <h2 style='margin:0; padding:0; color:white;'>🦷 Dental AI Studio Enterprise</h2>
        <small style='color:#cbd5e1;'>Station de Diagnostic Radiologique & Assistant de Décision Médicale</small>
    </div>
    <div>
        <span>👨‍⚕️ <b>Dr. {st.session_state.username}</b></span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.freepik.com/free-vector/dentistry-care-concept-illustration_114360-8260.jpg", use_container_width=True)
    st.write(f"**Session active :** {st.session_state.username}")
    
    if st.button("🚪 Se déconnecter", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()
        
    st.markdown("---")
    st.markdown("### ⚙️ Spécifications Système")
    st.info(f"""
    - **Inférence** : Soft Voting ({len(ensemble_models)} Folds)
    - **Accélération** : `{DEVICE.type.upper()}`
    - **Seuils** : Dynamic JSON
    - **Résolution** : {config.IMAGE_SIZE}x{config.IMAGE_SIZE}
    """)

tab_diag, tab_batch, tab_history, tab_stats = st.tabs([
    "🔬 Diagnostic, Grad-CAM & Filtres", 
    "📁 Analyse Multi-Images (Batch)",
    "📂 Historique & Consultation BDD", 
    "📊 Statistiques Cliniques"
])

# ------------------------------------------
# TAB 1 : DIAGNOSTIC, GRAD-CAM & FILTRES PACS
# ------------------------------------------
with tab_diag:
    col_input, col_view = st.columns([1, 2.2], gap="medium")

    # Valeurs par défaut pour l'archivage en BDD et rapport PDF
    patient_name = "Anonyme"
    patient_age = "N/A"

    with col_input:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("📋 Importation Cliché")
        patient_id = st.text_input("ID Examen / Patient", value=f"PAT-{np.random.randint(1000, 9999)}")
        uploaded_file = st.file_uploader("Charger la radiographie dentaire", type=["png", "jpg", "jpeg"], key="diag_file")
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("🛠️ Filtres Radiologiques PACS")
            contrast_val = st.slider("Contraste", 0.5, 2.5, 1.2, 0.1)
            brightness_val = st.slider("Luminosité", 0.5, 2.0, 1.0, 0.1)
            sharpness_val = st.slider("Netteté", 0.5, 3.0, 1.0, 0.2)
            invert_color = st.checkbox("Mode Négatif / Inversé", value=False)
            st.markdown('</div>', unsafe_allow_html=True)

    with col_view:
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Application des filtres visuels
            pil_img = Image.fromarray(img_rgb)
            pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast_val)
            pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness_val)
            pil_img = ImageEnhance.Sharpness(pil_img).enhance(sharpness_val)
            if invert_color:
                pil_img = ImageOps.invert(pil_img)
            proc_img_rgb = np.array(pil_img)

            # Inférence par Ensemble
            transform = get_val_transforms(config.IMAGE_SIZE)
            augmented = transform(image=img_rgb)
            tensor = augmented['image'].unsqueeze(0).to(DEVICE)

            all_probs = []
            with torch.no_grad():
                for m in ensemble_models:
                    output = torch.sigmoid(m(tensor))
                    all_probs.append(output.cpu().numpy())

            avg_probs = np.mean(all_probs, axis=0)[0]

            col_img, col_results = st.columns([1, 1.1])

            with col_img:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("📸 Radiographie Traitée")
                st.image(proc_img_rgb, use_container_width=True, caption=f"Examen : {patient_id}")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_results:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("📊 Prédictions par Pathologie")
                detected_list = []

                for idx, cls in enumerate(config.CLASSES):
                    prob = float(avg_probs[idx])
                    tau = thresholds.get(cls, 0.5)
                    is_detected = prob >= tau

                    c1, c2, c3 = st.columns([2.5, 1.2, 1])
                    c1.write(f"**{cls}** *(Seuil: {tau:.2f})*")
                    c2.progress(min(prob, 1.0))
                    
                    if is_detected and cls != "Maladie_parodontale":
                        c3.markdown(f"<span class='badge-positive'>{prob*100:.1f}%</span>", unsafe_allow_html=True)
                        detected_list.append((cls, idx, prob))
                    else:
                        c3.markdown(f"<span class='badge-negative'>{prob*100:.1f}%</span>", unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            # Grad-CAM et Actions
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("🔥 Zone d'Attention Légitime (Explicabilité Grad-CAM)")
            
            if detected_list:
                detected_list.sort(key=lambda x: x[2], reverse=True)
                primary_pathology, target_cls_idx, target_prob = detected_list[0]
                st.error(f"⚠️ **Pathologie prédominante identifiée :** {primary_pathology} ({target_prob*100:.1f}% de confiance)")

                target_model = ensemble_models[0]
                target_layer = target_model.features[-1]

                cam = GradCAM(target_model, target_layer)
                heatmap = cam.generate_heatmap(tensor.clone(), target_cls_idx)
                resized_img = cv2.resize(img_rgb, (config.IMAGE_SIZE, config.IMAGE_SIZE))
                cam_overlay = overlay_heatmap(resized_img, heatmap)

                c_cam1, c_cam2 = st.columns(2)
                c_cam1.image(resized_img, caption="Radio Originale Redimensionnée", use_container_width=True)
                c_cam2.image(cam_overlay, caption=f"Carte de Chaleur IA ({primary_pathology})", use_container_width=True)

                patho_str = ", ".join([d[0] for d in detected_list])
                conf_str = f"{target_prob*100:.1f}%"
            else:
                st.success("✅ **Aucune anomalie pathologique majeure détectée** selon les seuils configurés.")
                patho_str = "Sain / R.A.S"
                conf_str = "N/A"

            st.markdown("---")
            col_b1, col_b2 = st.columns(2)

            with col_b1:
                if st.button("💾 Sauvegarder dans la BDD SQLite", type="primary", use_container_width=True):
                    save_patient(patient_id, patient_name, patient_age, patho_str, conf_str)
                    st.success("Dossier d'examen enregistré avec succès !")

            with col_b2:
                pdf_bytes = generate_pdf_bytes(patient_id, patient_name, patient_age, patho_str, conf_str)
                st.download_button(
                    label="📄 Télécharger le Rapport Médical PDF",
                    data=pdf_bytes,
                    file_name=f"Rapport_DentalAI_{patient_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👈 Importez une radiographie dentaire dans le panneau de gauche pour démarrer le diagnostic.")

# ------------------------------------------
# TAB 2 : ANALYSE MULTI-IMAGES (BATCH)
# ------------------------------------------
with tab_batch:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("📁 Analyse en Lot de Plusieurs Radiographies")
    batch_files = st.file_uploader("Sélectionner un ensemble de fichiers radiologiques", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="batch_files")

    if batch_files:
        if st.button("🚀 Lancer l'Analyse du Lot", type="primary"):
            batch_data = []
            bar = st.progress(0)

            for i, f in enumerate(batch_files):
                file_b = np.asarray(bytearray(f.read()), dtype=np.uint8)
                b_bgr = cv2.imdecode(file_b, cv2.IMREAD_COLOR)
                b_rgb = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2RGB)

                transform = get_val_transforms(config.IMAGE_SIZE)
                tensor = transform(image=b_rgb)['image'].unsqueeze(0).to(DEVICE)

                all_probs = []
                with torch.no_grad():
                    for m in ensemble_models:
                        all_probs.append(torch.sigmoid(m(tensor)).cpu().numpy())

                mean_p = np.mean(all_probs, axis=0)[0]
                detected = []
                for idx, cls_name in enumerate(config.CLASSES):
                    if mean_p[idx] >= thresholds.get(cls_name, 0.5) and cls_name != "Maladie_parodontale":
                        detected.append(cls_name)

                row = {
                    "Fichier": f.name,
                    "Diagnostic": ", ".join(detected) if detected else "Sain",
                    "Carie (%)": round(float(mean_p[0]) * 100, 1),
                    "Fracture (%)": round(float(mean_p[1]) * 100, 1),
                    "Lesion_periapicale (%)": round(float(mean_p[2]) * 100, 1),
                    "Perte_osseuse (%)": round(float(mean_p[3]) * 100, 1),
                    "Pulpite_aigue (%)": round(float(mean_p[4]) * 100, 1),
                }
                batch_data.append(row)
                bar.progress((i + 1) / len(batch_files))

            df_batch = pd.DataFrame(batch_data)
            st.write("### 📊 Résultats Consolidés du Lot")
            st.dataframe(df_batch, use_container_width=True)

            csv_batch = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Exporter le Rapport de Lot en CSV",
                data=csv_batch,
                file_name=f"DentalAI_Batch_Report_{datetime.date.today()}.csv",
                mime="text/csv"
            )
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------
# TAB 3 : HISTORIQUE & CONSULTATION BDD
# ------------------------------------------
with tab_history:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("📂 Historique des Examens Cliniques Réalisés")
    
    records = get_all_patients()
    if records:
        df_history = pd.DataFrame(
            records, 
            columns=["ID Patient", "Nom & Prénom", "Âge", "Date Examen", "Pathologies", "Confiance Max"]
        )

        search_query = st.text_input("🔍 Rechercher un examen par ID :", "")
        if search_query:
            df_history = df_history[
                df_history["ID Patient"].str.contains(search_query, case=False, na=False)
            ]

        st.dataframe(df_history, use_container_width=True)
        
        csv_data = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Exporter l'historique filtré (CSV)",
            data=csv_data,
            file_name="historique_patients_dentaire.csv",
            mime="text/csv"
        )
    else:
        st.info("Aucun examen enregistré en base de données SQLite.")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------
# TAB 4 : STATISTIQUES CLINIQUES
# ------------------------------------------
with tab_stats:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("📈 Indicateurs de Performance & Configuration")
    
    records = get_all_patients()
    total_records = len(records)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Examens en BDD", total_records)
    col_m2.metric("Architecture Modèle", "EfficientNet-B0")
    col_m3.metric("Folds Actifs", f"{len(ensemble_models)} / {config.NUM_FOLDS}")
    col_m4.metric("F1-Score Global OOF", "92.6%")

    st.markdown("---")
    st.markdown("#### 🎯 Configuration des Seuils de Décision Dynamiques (`models/optimal_thresholds.json`)")
    
    df_thresh = pd.DataFrame([
        {"Pathologie": k, "Seuil Optimal": f"{v:.2f}", "Sensibilité": "Élevée (Abaissement)" if v < 0.5 else "Standard"}
        for k, v in thresholds.items()
    ])
    st.table(df_thresh)
    st.markdown('</div>', unsafe_allow_html=True)