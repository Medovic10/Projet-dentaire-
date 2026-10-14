import os
import torch

# Désactive la recherche de mise à jour d'Albumentations pour éviter le timeout
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Recherche dynamique du fichier de métadonnées
possible_metadata = [
    os.path.join(BASE_DIR, "patients_preprocessed.csv"),
    os.path.join(BASE_DIR, "patients_diagnostics.csv"),
    os.path.join(BASE_DIR, "patients_diagnostics.xlsx"),
    os.path.join(BASE_DIR, "metadata", "patients_preprocessed.csv"),
    os.path.join(BASE_DIR, "metadata", "patients_diagnostics.csv"),
    r"D:\te\prj_dentaire\patients_preprocessed.csv",
    r"D:\te\prj_dentaire\patients_diagnostics.csv"
]

METADATA_PATH = None
for path in possible_metadata:
    if os.path.exists(path):
        METADATA_PATH = path
        break

if METADATA_PATH is None:
    raise FileNotFoundError("❌ Copiez 'patients_preprocessed.csv' directement dans D:\\te\\prjdent\\")

# Recherche du dossier contenant les images radiographiques
possible_data_dirs = [
    os.path.join(BASE_DIR, "data"),
    r"D:\te\PRJ\PRJ",
    r"D:\te\prj_dentaire\data"
]

DATA_DIR = next((d for d in possible_data_dirs if os.path.exists(d)), possible_data_dirs[0])
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CLASSES = [
    "Normal",
    "Carie",
    "Fracture",
    "Lesion_periapicale",
    "Perte_osseuse",
    "Pulpite_aigue",
    "Maladie_parodontale"
]

NUM_CLASSES = len(CLASSES)
NUM_FOLDS = 5
BATCH_SIZE = 16
EPOCHS = 5
LR = 3e-4
WEIGHT_DECAY = 1e-3
IMAGE_SIZE = 224
DROPOUT = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"📄 Métadonnées : {METADATA_PATH}")
print(f"📁 Dossier images : {DATA_DIR}")