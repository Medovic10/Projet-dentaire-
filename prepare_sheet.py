import os
import csv
import re

DATASET_PATH = r"D:\te\PRJ\PRJ"
OUTPUT_CSV = r"D:\te\prj_dentaire\patients_diagnostics.csv"

def natural_sort_key(s):
    """
    Découpe une chaîne de caractères en morceaux de textes et de nombres.
    Permet un tri naturel (PATIENT 1, PATIENT 2 ... PATIENT 10) sans planter.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def generate_empty_sheet(base_path, output_file):
    if not os.path.exists(base_path):
        print(f"❌ Chemin de la base de données introuvable : {base_path}")
        return

    # Récupérer uniquement les dossiers patients
    patients = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # Tri naturel robuste
    patients.sort(key=natural_sort_key)

    print(f"📋 Génération de la feuille de saisie pour {len(patients)} patients...")

    # Écriture du fichier CSV
    with open(output_file, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=';')
        # En-tête du fichier
        writer.writerow(["Dossier_Patient", "Nombre_Radios", "Diagnostic_Principal", "Notes_Cliniques"])
        
        for patient in patients:
            folder_path = os.path.join(base_path, patient)
            # Compter les images dans ce dossier
            image_count = len([img for img in os.listdir(folder_path) if os.path.splitext(img)[1].lower() in ['.jpg', '.jpeg', '.png']])
            
            # Écrire une ligne vide pour la saisie des diagnostics
            writer.writerow([patient, image_count, "", ""])

    print(f"🎉 Fichier créé avec succès : {output_file}")
    print("👉 Tu peux maintenant l'ouvrir directement dans Excel ou Notepad++ pour le remplir !")

if __name__ == "__main__":
    generate_empty_sheet(DATASET_PATH, OUTPUT_CSV)