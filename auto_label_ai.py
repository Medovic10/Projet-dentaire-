import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np

CSV_PATH = "data/dataset_metadata.csv"
CLASSES = ["Carie", "Fracture", "Lesion_periapicale", "Perte_osseuse", "Pulpite_aigue", "Maladie_parodontale"]

if not os.path.exists(CSV_PATH):
    print("❌ Fichier metadata introuvable.")
    exit()

df = pd.read_csv(CSV_PATH)

# Assurer l'existence des colonnes
for col in CLASSES + ["Normal"]:
    if col not in df.columns:
        df[col] = 0

# Séparation des données : Annotées vs Non annotées
labeled_mask = (df[CLASSES].sum(axis=1) > 0) | (df['Normal'] == 1)
df_labeled = df[labeled_mask].copy()
df_unlabeled = df[~labeled_mask].copy()

print(f"📊 Images déjà annotées (Train) : {len(df_labeled)}")
print(f"🔮 Images à annoter par l'IA : {len(df_unlabeled)}")

if len(df_unlabeled) == 0:
    print("✅ Toutes les images sont déjà annotées !")
    exit()

# Dataset PyTorch basique
class SimpleDentalDataset(Dataset):
    def __init__(self, dataframe, transform=None, is_labeled=True):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.is_labeled = is_labeled

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

        if self.is_labeled:
            labels = row[CLASSES].values.astype(np.float32)
            return image, torch.tensor(labels)
        return image, row['image_path']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader = DataLoader(SimpleDentalDataset(df_labeled, transform=transform), batch_size=16, shuffle=True)
unlabeled_loader = DataLoader(SimpleDentalDataset(df_unlabeled, transform=transform, is_labeled=False), batch_size=16, shuffle=False)

# 1. Modèle pré-entraîné (MobileNetV2 rapide)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Entraînement sur : {device}")

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 2. Entraînement rapide sur 8 époques
model.train()
for epoch in range(8):
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("✅ Modèle temporaire entraîné avec succès !")

# 3. Prédiction automatique sur les 595 images restantes
model.eval()
predictions = []

with torch.no_grad():
    for imgs, paths in unlabeled_loader:
        imgs = imgs.to(device)
        outputs = torch.sigmoid(model(imgs))
        probs = outputs.cpu().numpy()
        
        for path, prob in zip(paths, probs):
            # Seuil de décision = 0.40
            pred_classes = (prob > 0.40).astype(int)
            is_normal = 1 if np.sum(pred_classes) == 0 else 0
            
            row_dict = {'image_path': path, 'Normal': is_normal}
            for cls_idx, cls_name in enumerate(CLASSES):
                row_dict[cls_name] = pred_classes[cls_idx]
            predictions.append(row_dict)

# 4. Mise à jour automatique du CSV
df_preds = pd.DataFrame(predictions)

for _, pred_row in df_preds.iterrows():
    idx = df[df['image_path'] == pred_row['image_path']].index
    if len(idx) > 0:
        df.loc[idx[0], 'Normal'] = pred_row['Normal']
        for cls in CLASSES:
            df.loc[idx[0], cls] = pred_row[cls]

df.to_csv(CSV_PATH, index=False)
print(f"\n🎉 SUCCÈS : {len(df_unlabeled)} images ont été automatiquement annotées par l'IA !")