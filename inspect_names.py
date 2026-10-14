import os

DATASET_PATH = r"D:\te\PRJ\PRJ"

def inspect_structure(base_path):
    if not os.path.exists(base_path):
        print("❌ Chemin introuvable.")
        return

    print("==================================================")
    print("🔍 INSPECTION DE LA STRUCTURE DES LABELS")
    print("==================================================")

    # Récupérer les 3 premiers dossiers patients pour échantillonner
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    print(f"Exemple de noms de dossiers patients (total {len(subdirs)}) :")
    for subdir in subdirs[:5]:
        print(f"  📁 {subdir}")
    print("  ... \n")

    print("Exemple de fichiers à l'intérieur de ces dossiers :")
    for subdir in subdirs[:3]:
        folder_path = os.path.join(base_path, subdir)
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        print(f"  📁 Dossier : {subdir}")
        if files:
            for file in files[:4]:  # Affiche les 4 premiers fichiers
                print(f"    📄 {file}")
            if len(files) > 4:
                print(f"    ... (+ {len(files) - 4} autres fichiers)")
        else:
            print("    ⚠️ (Dossier vide)")
        print()

if __name__ == "__main__":
    inspect_structure(DATASET_PATH)