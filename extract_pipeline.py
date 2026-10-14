import fitz  # PyMuPDF
import re
import os
import pandas as pd
import hashlib

# Configuration des chemins
RAW_DIR = "data/01_raw"
EXTRACTED_AVANT_DIR = "data/02_extracted_images/avant"
EXTRACTED_APRES_DIR = "data/02_extracted_images/apres"
EXTRACTED_INTER_DIR = "data/02_extracted_images/intermediaire"
CSV_PATH = "metadata/dataset_initial.csv"

# Création des dossiers
os.makedirs(EXTRACTED_AVANT_DIR, exist_ok=True)
os.makedirs(EXTRACTED_APRES_DIR, exist_ok=True)
os.makedirs(EXTRACTED_INTER_DIR, exist_ok=True)

def anonymize_patient_id(dossier_str):
    return hashlib.md5(dossier_str.encode('utf-8')).hexdigest()[:8]

def process_pdf(pdf_path, filename):
    doc = fitz.open(pdf_path)
    text_content = ""
    valid_radios = []
    
    # 1. Extraction et double filtrage (Taille + Forme)
    for page in doc:
        text_content += page.get_text("text")
        image_list = page.get_images(full=True)
        
        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            
            width = base_image["width"]
            height = base_image["height"]
            surface = width * height
            
            # Calcul de la forme de l'image (pour bloquer les barres noires)
            if height > 0:
                aspect_ratio = width / height
            else:
                aspect_ratio = 0
            
            # RÈGLE : La surface doit être assez grande ET la forme doit ressembler à une photo (ratio entre 0.4 et 2.5)
            if surface > 50000 and 0.4 < aspect_ratio < 2.5:
                valid_radios.append(base_image)

    print(f"[{filename}] -> {len(valid_radios)} vraie(s) radio(s) trouvée(s).")

    # 2. Parsing des métadonnées
    dossier_match = re.search(r"No\.?\s*de\s*dossier\s*:\s*(\d+_\d+)", text_content, re.IGNORECASE)
    id_dossier = dossier_match.group(1) if dossier_match else None
    
    sexe_match = re.search(r"Sexe\s*:\s*(\w+)", text_content, re.IGNORECASE)
    age_match = re.search(r"Age\s*:\s*(\d+)", text_content, re.IGNORECASE)
    
    sexe = sexe_match.group(1) if sexe_match else "Inconnu"
    age = age_match.group(1) if age_match else "Inconnu"

    if id_dossier:
        patient_id = f"PAT_{anonymize_patient_id(id_dossier)}"
    else:
        safe_name = os.path.splitext(filename)[0].replace(" ", "_")
        patient_id = f"FILE_{safe_name}"

    # 3. Sauvegarde dynamique
    path_avant = ""
    path_apres = ""
    paths_inter_list = []
    
    if len(valid_radios) == 1:
        path_avant = os.path.join(EXTRACTED_AVANT_DIR, f"{patient_id}_unique.{valid_radios[0]['ext']}")
        with open(path_avant, "wb") as f:
            f.write(valid_radios[0]["image"])
            
    elif len(valid_radios) >= 2:
        path_avant = os.path.join(EXTRACTED_AVANT_DIR, f"{patient_id}_avant.{valid_radios[0]['ext']}")
        path_apres = os.path.join(EXTRACTED_APRES_DIR, f"{patient_id}_apres.{valid_radios[-1]['ext']}")
        
        with open(path_avant, "wb") as f:
            f.write(valid_radios[0]["image"])
        with open(path_apres, "wb") as f:
            f.write(valid_radios[-1]["image"])
            
        for i in range(1, len(valid_radios) - 1):
            path_inter = os.path.join(EXTRACTED_INTER_DIR, f"{patient_id}_inter_{i}.{valid_radios[i]['ext']}")
            with open(path_inter, "wb") as f:
                f.write(valid_radios[i]["image"])
            paths_inter_list.append(path_inter)

    doc.close()
    
    return {
        "id_patient": patient_id,
        "age": age,
        "sexe": sexe,
        "image_avant_path": path_avant,
        "image_apres_path": path_apres,
        "images_intermediaires": " | ".join(paths_inter_list),
        "symptomes_initiaux": "",
        "diagnostic": "",
        "traitement_realise": ""
    }

def build_dataset():
    data = []
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(RAW_DIR, filename)
            row_data = process_pdf(pdf_path, filename)
            data.append(row_data)
            
    df = pd.DataFrame(data)
    df.to_csv(CSV_PATH, index=False, sep=';')
    print(f"\n--- Bilan ---")
    print(f"Extraction terminée. Filtrage géométrique appliqué.")

if __name__ == "__main__":
    build_dataset()