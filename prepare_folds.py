import os
import pandas as pd
from sklearn.model_selection import KFold

CSV_IN = "data/dataset_metadata.csv"
CSV_OUT = "data/dataset_5folds.csv"

if not os.path.exists(CSV_IN):
    print(f"❌ Fichier {CSV_IN} introuvable.")
    exit()

df = pd.read_csv(CSV_IN)

# Séparation des 735 images en 5 plis (folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
df['fold'] = -1

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    df.loc[val_idx, 'fold'] = fold

df.to_csv(CSV_OUT, index=False)
print(f"✅ Fichier {CSV_OUT} généré avec succès ({len(df)} images réparties sur 5 Folds).")