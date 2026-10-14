import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

# --- CONFIGURATION ---
MODEL_PATH = r"D:\te\prj_dentaire\best_dental_efficientnet.pth"
CLASSES = [
    "Normal", 
    "Carie", 
    "Fracture", 
    "Lesion_periapicale", 
    "Perte_osseuse", 
    "Pulpite_aigue", 
    "Maladie_parodontale"
]

OBSERVATIONS_DICT = {
    "Normal": "Examen clinique normal. Structures osseuses et dentaires physiologiques.",
    "Carie": "Perte de substance coronaire radiotransparente évoquant une lésion carieuse.",
    "Fracture": "Trait de fracture net détecté sur la structure dentaire.",
    "Lesion_periapicale": "Image radioclaire péri-apicale évocatrice d'une lésion endodontique.",
    "Perte_osseuse": "Résorption osseuse alvéolaire observée.",
    "Pulpite_aigue": "Complication pulpaire profonde nécessitant un traitement endodontique urgent.",
    "Maladie_parodontale": "Atteinte des tissus de soutien et lyse osseuse diffuse."
}

st.set_page_config(page_title="Dental AI - Assistant Diagnostic", layout="wide")

st.title("🦷 IA d'Aide au Diagnostic Radiologique Dentaire")
st.write("Importez une nouvelle radiographie pour obtenir une analyse multi-label instantanée et un rapport d'examen.")

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_trained_model():
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_features, len(CLASSES))
    )
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        st.error(f"Fichier modèle introuvable à l'emplacement : {MODEL_PATH}. Lancez d'abord 'train.py'.")
        return None

model = load_trained_model()

# Transformations image
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- INTERFACE UTILISATEUR ---
uploaded_file = st.file_uploader("Choisissez une radiographie dentaire (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.subheader("📸 Radiographie")
        st.image(image, caption="Image importée", use_container_width=True)
        
    with col2:
        st.subheader("📊 Prédictions Multi-Label")
        
        if model is not None:
            # Prétraitement et Inférence
            input_tensor = img_transforms(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.sigmoid(outputs).squeeze().numpy()
            
            detected_classes = []
            
            # Affichage des probabilités
            for cls_name, prob in zip(CLASSES, probs):
                percentage = float(prob) * 100
                st.write(f"**{cls_name}** : {percentage:.1f}%")
                st.progress(float(prob))
                
                # Seuil de détection fixé à 50%
                if prob >= 0.5:
                    detected_classes.append(cls_name)
            
            st.write("---")
            st.subheader("📋 Rapport d'Observation Automatique")
            
            if detected_classes:
                notes = []
                for diag in detected_classes:
                    obs = OBSERVATIONS_DICT.get(diag, "Anomalie détectée.")
                    notes.append(f"• **{diag}** : {obs}")
                
                rapport_text = "\n".join(notes)
                st.info(rapport_text)
            else:
                st.success("• **Normal** : Aucune pathologie majeure détectée au-dessus du seuil de confiance.")