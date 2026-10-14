import pandas as pd
import os

CSV_PATH = r"D:\te\prj_dentaire\patients_diagnostics.csv"
CLASSES = [
    "Normal", 
    "Carie", 
    "Fracture", 
    "Lesion_periapicale", 
    "Perte_osseuse", 
    "Pulpite_aigue", 
    "Maladie_parodontale"
]

# Charger le dataset complet
df = pd.read_csv(CSV_PATH, sep=';', encoding='utf-8-sig', keep_default_na=False)

print("📊 Encodage Multi-Label des diagnostics en cours...")

# Initialiser les colonnes de chaque classe à 0
for cls in CLASSES:
    df[cls] = 0

# Appliquer le One-Hot Encoding multi-label
for idx, row in df.iterrows():
    diags = [d.strip() for d in str(row['Diagnostic_Principal']).split(",") if d.strip()]
    for diag in diags:
        if diag in CLASSES:
            df.at[idx, diag] = 1

# Sauvegarder une version prête pour l'entraînement
OUTPUT_CSV = r"D:\te\prj_dentaire\patients_preprocessed.csv"
df.to_csv(OUTPUT_CSV, sep=';', index=False, encoding='utf-8-sig')

print(f"📊 Dataset d'entraînement généré avec succès à l'emplacement : {OUTPUT_CSV}")
print("\nAperçu des premières lignes :")
print(df[['Dossier_Patient'] + CLASSES].head())