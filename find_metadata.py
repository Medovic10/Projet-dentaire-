import os

DATASET_PATH = r"D:\te\PRJ"

def find_metadata_files(base_path):
    print("==================================================")
    print("🔍 RECHERCHE DE DOCUMENTS DE DIAGNOSTICS (METADATA)")
    print("==================================================")
    
    extensions_recherchees = ['.csv', '.xlsx', '.xls', '.txt', '.json']
    fichiers_trouves = []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in extensions_recherchees:
                # Éviter de lister les fichiers d'images ou les scripts connus
                if "inspect_names" not in file and "explore_dataset" not in file:
                    chemin_complet = os.path.join(root, file)
                    fichiers_trouves.append((file, chemin_complet))

    if fichiers_trouves:
        print(f"🎉 Trouvé {len(fichiers_trouves)} fichier(s) suspect(s) qui pourrai(en)t contenir les étiquettes :\n")
        for nom, chemin in fichiers_trouves:
            print(f"  📄 Fichier : {nom}")
            print(f"     Chemin  : {chemin}\n")
    else:
        print("⚠️ Aucun fichier d'indexation (Excel, CSV, TXT) trouvé dans le dossier.")
        print("Il est possible que les diagnostics soient saisis d'une autre manière.")

if __name__ == "__main__":
    find_metadata_files(DATASET_PATH)