import torch
from torchvision import models, transforms
from PIL import Image
import os

# Étape 4.1 : Importation de notre générateur de rapports cliniques
from generate_report import ClinicalReportGenerator

# Configuration
MODEL_PATH = "models/resnet50_dentaire_final.pth"
CLASSES = ['Carie', 'Dent_fracturee', 'Lesion_periapicale', 'Normal', 'Perte_osseuse']

def load_model(model_path):
    # Reconstruire l'architecture ResNet-50
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 5)
    
    # Charger les poids entraînés
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(image_path, model):
    # Prétraitement de l'image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)  # Ajout de la dimension batch (1, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probabilities, 1)
        
    return CLASSES[pred.item()], conf.item()

if __name__ == "__main__":
    print("🤖 Initialisation du système d'analyse dentaire IA...")
    model = load_model(MODEL_PATH)
    
    # Gestion automatique de l'extension (.jpeg ou .jpg)
    possible_paths = ["data/test_radio.jpeg", "data/test_radio.jpg"]
    test_image = None
    
    for path in possible_paths:
        if os.path.exists(path):
            test_image = path
            break
    
    if test_image:
        print(f"📸 Analyse en cours pour l'image : {test_image}")
        label, conf = predict(test_image, model)
        
        print(f"\n--- 🔍 RÉSULTAT DE L'ANALYSE IA ---")
        print(f"Diagnostic détecté : {label}")
        print(f"Indice de confiance : {conf:.2%}")
        print(f"-----------------------------------\n")
        
        # --- ÉTAPE 4.1 : Génération automatique du rapport lié au diagnostic ---
        print("📝 Génération automatique du rapport de consultation...")
        generateur = ClinicalReportGenerator()
        
        # Métadonnées fictives du patient (que l'on pourra lier à une interface graphique)
        patient_id = "PAT_9942"
        symptomes_observes = "Légère sensibilité à la pression signalée par le patient."
        historique_soins = "Détartrage effectué il y a 6 mois."
        
        # Génération et sauvegarde automatique du rapport
        # Note : on multiplie 'conf' par 100 pour l'affichage en pourcentage dans le rapport
        rapport = generateur.create_text_report(
            patient_id=patient_id,
            diagnostic=label,
            confiance=conf * 100,
            symptomes=symptomes_observes,
            traitement=historique_soins,
            save_to_file=True
        )
        
        print("\n================ APERÇU DU RAPPORT CRÉÉ ================")
        print(rapport)
        print("========================================================")
        
    else:
        print(f"⚠️ Aucun fichier test_radio.jpeg ou .jpg détecté dans le dossier 'data/'.")