import os
import pandas as pd
import config

# Chemins de configuration
DATA_DIR = "data"
CSV_PATH = "data/dataset_metadata.csv"
CLASSES = getattr(config, 'CLASSES', [
    "Carie", "Fracture", "Lesion_periapicale", 
    "Perte_osseuse", "Pulpite_aigue", "Maladie_parodontale"
])

print(f"🔍 Recherche récursive d'images dans : {os.path.abspath(DATA_DIR)} ...")

# Extensions supportées
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jpg', '.jpeg', '.png')
image_files = []

# Parcours récursif de tous les sous-dossiers dans data/
for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.lower().endswith(valid_extensions):
            # Formater le chemin pour éviter les problèmes sous Windows
            full_path = os.path.join(root, file).replace("\\", "/")
            image_files.append(full_path)

print(f"📷 Nombre d'images trouvées : {len(image_files)}")

if len(image_files) == 0:
    print("⚠️ Aucune image trouvée ! Assurez-vous que vos images (.jpg, .png) sont déposées dans 'data/' ou l'un de ses sous-dossiers.")
else:
    # Si le CSV existe déjà, on met à jour sans écraser les labels existants
    if os.path.exists(CSV_PATH):
        df_existing = pd.read_csv(CSV_PATH)
        existing_paths = set(df_existing['image_path'].tolist()) if 'image_path' in df_existing.columns else set()
        
        new_images = [img for img in image_files if img not in existing_paths]
        
        if new_images:
            df_new = pd.DataFrame({'image_path': new_images})
            for cls in CLASSES:
                df_new[cls] = 0  # Initialiser à 0
            
            df_updated = pd.concat([df_existing, df_new], ignore_index=True)
            df_updated.to_csv(CSV_PATH, index=False)
            print(f"✅ {len(new_images)} nouvelles images ajoutées au fichier {CSV_PATH}.")
        else:
            print("ℹ️ Toutes les images trouvées sont déjà recensées dans le CSV.")
    else:
        # Création initiale du CSV
        df = pd.DataFrame({'image_path': image_files})
        for cls in CLASSES:
            df[cls] = 0
            
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        df.to_csv(CSV_PATH, index=False)
        print(f"✅ Fichier {CSV_PATH} créé avec succès avec {len(image_files)} images.")