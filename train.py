import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np

# Configuration
CSV_PATH = "data/dataset_5folds.csv"
MODEL_DIR = "models"
CLASSES = ["Carie", "Fracture", "Lesion_periapicale", "Perte_osseuse", "Pulpite_aigue", "Maladie_parodontale"]
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    print(f"❌ Fichier {CSV_PATH} introuvable. Exécutez d'abord prepare_folds.py.")
    exit()

df_all = pd.read_csv(CSV_PATH)

# Dataset PyTorch
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

# Transforms avec Data Augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Apprentissage lancé sur : {device}")

# Entraînement des 5 Folds (EfficientNet-B0)
for fold in range(5):
    print(f"\n==================== Fold {fold + 1}/5 ====================")
    df_train = df_all[df_all['fold'] != fold]
    df_val = df_all[df_all['fold'] == fold]

    train_loader = DataLoader(DentalDataset(df_train, train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(DentalDataset(df_val, val_transform), batch_size=BATCH_SIZE, shuffle=False)

    # Modèle EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(df_train)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(df_val)

        print(f"Époque {epoch+1:02d}/{EPOCHS:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(MODEL_DIR, f"efficientnet_fold{fold}.pth")
            torch.save(model.state_dict(), save_path)

    print(f"✅ Meilleur modèle Fold {fold+1} sauvegardé sous {MODEL_DIR}/efficientnet_fold{fold}.pth")

print("\n🎉 Entraînement des 5 modèles terminé avec succès !")