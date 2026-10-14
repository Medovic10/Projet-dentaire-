import os
import re
import pandas as pd

CSV_PATH = "data/dataset_metadata.csv"
PATIENTS_CSV = "patients_preprocessed.csv"

CLASSES = [
    "Carie", "Fracture", "Lesion_periapicale", 
    "Perte_osseuse", "Pulpite_aigue", "Maladie_parodontale"
]

if not os.path.exists(CSV_PATH) or not os.path.exists(PATIENTS_CSV):
    print("❌ Fichier dataset_metadata.csv ou patients_preprocessed.csv introuvable.")
    exit()

df_meta = pd.read_csv(CSV_PATH)
df_pat = pd.read_csv(PATIENTS_CSV, sep=';', engine='python', on_bad_lines='skip')

# Normalisation des colonnes de patients_preprocessed.csv
df_pat.columns = [c.strip().lower() for c in df_pat.columns]

print(f"📊 Colonnes détectées dans {PATIENTS_CSV} : {list(df_pat.columns)}")

synced_count = 0

for idx, row in df_meta.iterrows():
    path_str = str(row['image_path'])
    
    # Extraction du nom ou numéro de patient (ex: 'PATIENT 32' ou '32')
    match_patient = re.search(r'PATIENT[\s_]*(\d+)', path_str, re.IGNORECASE)
    
    if match_patient:
        patient_num = match_patient.group(1) # Ex: "32"
        
        # Recherche dans patients_preprocessed.csv
        # Cherche si le numéro '32' ou 'PATIENT 32' est présent dans une des colonnes ID/Patient
        match_row = None
        for col in df_pat.columns:
            matches = df_pat[df_pat[col].astype(str).str.contains(rf'\b{patient_num}\b', regex=True, na=False)]
            if not matches.empty:
                match_row = matches.iloc[0]
                break
        
        if match_row is not None:
            # Transfert des labels
            for cls in CLASSES:
                cls_lower = cls.lower()
                for pat_col in df_pat.columns:
                    if cls_lower in pat_col:
                        val = match_row[pat_col]
                        if str(val).strip() in ['1', '1.0', 'True', 'true']:
                            df_meta.at[idx, cls] = 1
            synced_count += 1

df_meta.to_csv(CSV_PATH, index=False)
print(f"\n✅ {synced_count} / {len(df_meta)} images synchronisées par ID Patient !")