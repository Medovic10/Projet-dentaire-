import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, multilabel_confusion_matrix

CSV_PATH = "data/dataset_5folds.csv"
MODEL_DIR = "models"
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
print(f"⚡ Évaluation Out-Of-Fold sur : {device}")

oof_targets = []
oof_predictions = []

for fold in range(5):
    model_path = os.path.join(MODEL_DIR, f"efficientnet_fold{fold}.pth")
    if not os.path.exists(model_path):
        print(f"⚠️ Checkpoint introuvable : {model_path}")
        continue

    df_val = df_all[df_all['fold'] == fold]
    val_loader = DataLoader(DentalDataset(df_val, val_transform), batch_size=BATCH_SIZE, shuffle=False)

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    fold_preds = []
    fold_targets = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # Seuil de décision multi-label standard = 0.50
            preds = (probs > 0.50).astype(int)
            
            fold_preds.append(preds)
            fold_targets.append(labels.numpy())

    oof_predictions.append(np.vstack(fold_preds))
    oof_targets.append(np.vstack(fold_targets))

oof_predictions = np.vstack(oof_predictions)
oof_targets = np.vstack(oof_targets)

# 1. Rapport de Classification
print("\n" + "="*65)
print("📊 RAPPORT DE CLASSIFICATION OUT-OF-FOLD (735 RADIOGRAPHIES)")
print("="*65)
print(classification_report(oof_targets, oof_predictions, target_names=CLASSES, zero_division=0, digits=4))

# 2. Génération des Matrices de Confusion par Pathologie
mcm = multilabel_confusion_matrix(oof_targets, oof_predictions)

fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=300)
axes = axes.flatten()

for i, cls_name in enumerate(CLASSES):
    cm = mcm[i]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                xticklabels=['Négatif', 'Positif'], yticklabels=['Négatif', 'Positif'])
    axes[i].set_title(f'Pathologie : {cls_name}', fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Prédiction')
    axes[i].set_ylabel('Réalité')

plt.tight_layout()
plt.savefig('oof_multilabel_confusion_matrices.png')
print("🖼️ Graphiques enregistrés sous 'oof_multilabel_confusion_matrices.png'.")