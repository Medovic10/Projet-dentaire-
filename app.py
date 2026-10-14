import streamlit as st
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
import torch.nn.functional as F

# Force la recharge du module pour éviter les problèmes de cache Streamlit
import importlib
import generate_report
importlib.reload(generate_report)
from generate_report import ClinicalReportGenerator

# --- CONFIGURATION DE LA PAGE STREAMLIT ---
st.set_page_config(
    page_title="Cabinet Dentaire IA - Assistant Clinique",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONFIGURATION DES DOSSIERS ET PARAMÈTRES ---
MODEL_PATH = "models/resnet50_dentaire_final.pth"
CLASSES = ['Carie', 'Dent_fracturee', 'Lesion_periapicale', 'Normal', 'Perte_osseuse']

# --- CHARGEMENT DU MODÈLE (MIS EN CACHE POUR ÉVITER DE RECHARGER À CHAQUE CLIC) ---
@st.cache_resource
def load_dental_model(model_path):
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 5)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# --- FONCTION DE PRÉDICTION ET EXTRACTION DES CONVOLUTIONS (GRAD-CAM) ---
def analyze_image(image_pil, model):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = preprocess(image_pil.convert('RGB')).unsqueeze(0)
    
    # Hook pour intercepter les cartes d'activation de la dernière couche de convolution (layer4)
    features = []
    def hook_fn(module, input, output):
        features.append(output)
    
    handle = model.layer4.register_forward_hook(hook_fn)
    
    # Inférence
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probabilities, 1)
    
    handle.remove()  # Nettoyage du hook
    
    # Calcul de la Heatmap
    activations = features[0].detach().squeeze()
    weights = model.fc.weight[pred].detach()
    heatmap = torch.matmul(weights, activations.view(activations.size(0), -1)).view(activations.size(1), activations.size(2))
    
    heatmap = F.relu(heatmap).numpy()
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
    return CLASSES[pred.item()], conf.item(), heatmap

# --- DESIGN DE L'INTERFACE ---
st.title("🦷 Cabinet Dentaire Connecté - Assistant IA")
st.write("Outil d'aide au diagnostic radiologique et d'édition automatique de comptes-rendus cliniques (Prototype PFA).")

# --- BARRE LATÉRALE : FORMULAIRE CLINIQUE ---
st.sidebar.header("📋 Dossier Patient")
patient_id = st.sidebar.text_input("Identifiant Unique du Patient", value="PAT-2026-99")
symptomes = st.sidebar.text_area(
    "Symptômes cliniques rapportés", 
    placeholder="Ex: Douleur aiguë molaire inférieure droite, sensibilité thermique..."
)
historique = st.sidebar.text_area(
    "Historique des soins", 
    placeholder="Ex: Pose de composite sur la 46 il y a 1 an..."
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Les informations saisies ici seront automatiquement intégrées au compte-rendu généré par l'IA.")

# --- PAGE PRINCIPALE : CHARGEMENT DE LA RADIO ---
uploaded_file = st.file_uploader("📂 Déposez la radiographie dentaire du patient (Format JPEG, JPG ou PNG)", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    # Charger l'image avec PIL
    image = Image.open(uploaded_file)
    
    # Création de deux colonnes principales pour l'affichage
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("📷 Image Source")
        st.image(image, use_container_width=True, caption="Radiographie originale")
        
    with col2:
        st.subheader("⚙️ Analyse IA")
        
        # Bouton d'activation de l'analyse
        if st.button("🔍 Lancer l'Analyse Diagnostique", type="primary"):
            with st.spinner("Analyse de l'image par ResNet-50 et génération de la carte d'activation..."):
                
                # Chargement du modèle et exécution
                model = load_dental_model(MODEL_PATH)
                label, conf, heatmap = analyze_image(image, model)
                
                # 1. Affichage des résultats numériques
                st.success(f"**Diagnostic détecté : {label.replace('_', ' ')}**")
                st.metric(label="Indice de confiance", value=f"{conf:.2%}")
                
                # 2. Superposition de la Heatmap
                st.write("🎯 **Zone d'activation de l'anomalie (Grad-CAM) :**")
                
                # Redimensionnement de la heatmap pour correspondre à l'image
                heatmap_resized = np.array(Image.fromarray(heatmap).resize(image.size, Image.Resampling.LANCZOS))
                masked_heatmap = np.ma.masked_where(heatmap_resized < 0.4, heatmap_resized)
                
                # Génération de la figure matplotlib
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(image.convert('L'), cmap='gray')
                ax.imshow(masked_heatmap, cmap='jet', alpha=0.55, vmin=0, vmax=1)
                ax.axis('off')
                
                # Affichage dans Streamlit
                st.pyplot(fig)
                
 # 3. Génération du rapport
                st.subheader("📝 Compte-rendu Clinique Généré")
                
                generateur = ClinicalReportGenerator()
                rapport_texte = generateur.create_text_report(
                    patient_id=patient_id,
                    diagnostic=label,
                    confiance=conf * 100,
                    symptomes=symptomes if symptomes else "Aucun symptôme renseigné.",
                    traitement=historique if historique else "Aucun antécédent renseigné.",
                    save_to_file=False # Pas de fichier physique temporaire côté serveur
                )
                
                # Affichage du rapport brut dans la zone de texte
                st.text_area("Aperçu du document brut", value=rapport_texte, height=250)
                
                # --- NOUVEAUTÉ : Génération du document PDF en mémoire ---
                pdf_bytes = generateur.create_pdf_report(
                    patient_id=patient_id,
                    diagnostic=label,
                    confiance=conf * 100,
                    symptomes=symptomes if symptomes else "Aucun symptôme renseigné.",
                    traitement=historique if historique else "Aucun antécédent renseigné."
                )
                
                # Alignement horizontal des boutons de téléchargement
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    st.download_button(
                        label="📄 Télécharger le compte-rendu (.txt)",
                        data=rapport_texte,
                        file_name=f"Rapport_IA_{patient_id}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                    
                with col_dl2:
                    st.download_button(
                        label="🔴 Télécharger le PDF Clinique (.pdf)",
                        data=pdf_bytes,
                        file_name=f"Rapport_Officiel_IA_{patient_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )