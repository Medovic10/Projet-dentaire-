import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import f1_score

CSV_PATH = "data/dataset_5folds.csv"
MODEL_DIR = "models"
OUTPUT_JSON = "models/thresholds.json"
CLASSES = ["Carie", "Fracture", "Lesion_periapicale", "Perte_osseuse", "Pulpite_aigue", "Maladie_parodontale"]
BATCH_SIZE = 16

if not os.path.exists(CSV_PATH):
    print(f"❌ Fichier {CSV_PATH} introuvable.")
    exit()

df_all = pd.read_csv(CSV_PATH)

class DentalDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color=0)

        if self.transform:
            image = self.transform(image)

        labels = row[CLASSES].values.astype(np.float32)
        return image, torch.tensor(labels)

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Optimisation des seuils sur : {device}")

oof_targets = []
oof_probs = []

for fold in range(5):
    model_path = os.path.join(MODEL_DIR, f"efficientnet_fold{fold}.pth")
    if not os.path.exists(model_path):
        continue

    df_val = df_all[df_all['fold'] == fold]
    val_loader = DataLoader(DentalDataset(df_val, val_transform), batch_size=BATCH_SIZE, shuffle=False)

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    fold_probs = []
    fold_targets = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            fold_probs.append(probs)
            fold_targets.append(labels.numpy())

    oof_probs.append(np.vstack(fold_probs))
    oof_targets.append(np.vstack(fold_targets))

oof_probs = np.vstack(oof_probs)
oof_targets = np.vstack(oof_targets)

best_thresholds = {}
print("\n🎯 Recherche des seuils optimaux par classe (Maximisation F1-Score) :\n")

for idx, cls_name in enumerate(CLASSES):
    targets_cls = oof_targets[:, idx]
    probs_cls = oof_probs[:, idx]

    if np.sum(targets_cls) == 0:
        best_thresholds[cls_name] = 0.5
        print(f"⚠️ {cls_name:<20} | Seuil par défaut: 0.50 | Aucun exemple positif")
        continue

    best_thresh = 0.5
    best_f1 = 0.0

    for thresh in np.arange(0.10, 0.90, 0.02):
        preds = (probs_cls > thresh).astype(int)
        f1 = f1_score(targets_cls, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    best_thresholds[cls_name] = round(float(best_thresh), 2)
    print(f"✅ {cls_name:<20} | Seuil optimal: {best_thresh:.2f} | F1-Score Max: {best_f1:.4f}")

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(best_thresholds, f, indent=4)

print(f"\n💾 Seuils sauvegardés avec succès dans '{OUTPUT_JSON}'.")