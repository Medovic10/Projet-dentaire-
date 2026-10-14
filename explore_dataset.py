import os
from collections import Counter

# Chemin de ta base de données (extrait de ta capture d'écran)
DATASET_PATH = r"D:\te\PRJ\PRJ"

def analyze_dataset(base_path):
    if not os.path.exists(base_path):
        print(f"❌ Le dossier {base_path} n'existe pas. Vérifie le chemin !")
        return

    print("==================================================")
    print("🔍 ANALYSE DE LA BASE DE DONNÉES DENTAIRE")
    print("==================================================")
    print(f"Chemin analysé : {base_path}\n")

    patient_folders = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    image_count = 0
    other_count = 0
    other_extensions = Counter()
    patient_images = {}

    # Parcours récursif des dossiers
    for root, dirs, files in os.walk(base_path):
        # Le premier niveau contient la liste des dossiers patients
        if root == base_path:
            patient_folders = dirs
            continue
        
        # Récupère le nom du dossier patient actuel
        patient_name = os.path.basename(root)
        patient_images[patient_name] = 0
        
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_count += 1
                patient_images[patient_name] += 1
            else:
                other_count += 1
                if ext:
                    other_extensions[ext] += 1
                else:
                    other_extensions['sans_extension'] += 1

    # --- AFFICHAGE DES RÉSULTATS ---
    print(f"📁 Nombre de dossiers de patients détectés : {len(patient_folders)}")
    print(f"🖼️ Radiographies réelles (images exploitables) : {image_count}")
    print(f"📄 Autres fichiers détectés (non-images)      : {other_count}")
    
    if other_count > 0:
        print("\nDétail des fichiers non-images trouvés :")
        for ext, count in other_extensions.items():
            print(f"  - {ext} : {count} fichier(s)")

    # Statistiques de répartition
    if patient_images and len(patient_folders) > 0:
        counts = list(patient_images.values())
        valid_counts = [c for c in counts if c > 0]
        
        if valid_counts:
            print("\n📊 Statistiques de répartition par patient :")
            print(f"  - Moyenne de radios par patient : {sum(valid_counts)/len(valid_counts):.1f}")
            print(f"  - Minimum de radios pour un patient : {min(valid_counts)} (Patient: {min(patient_images, key=lambda k: patient_images[k] if patient_images[k] > 0 else float('inf'))})")
            print(f"  - Maximum de radios pour un patient : {max(valid_counts)} (Patient: {max(patient_images, key=patient_images.get)})")
        else:
            print("\n⚠️ Attention : Aucun dossier patient ne contient d'images lisibles.")

if __name__ == "__main__":
    analyze_dataset(DATASET_PATH)