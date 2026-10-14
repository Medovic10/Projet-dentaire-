import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

# ==========================================================
# Ajout du dossier courant au PATH
# ==========================================================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==========================================================
# Import du Dataset
# ==========================================================
from dataset_loader import (
    LongitudinalDentaireDataset,
    clinical_vision_transforms,
    custom_collate_fn
)

# ==========================================================
# Configuration
# ==========================================================
CSV_PATH = "metadata/dataset_test_fictif.csv"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "resnet50_dentaire_final.pth")

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# Device
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n🚀 Device utilisé : {device}")

# ==========================================================
# Dataset
# ==========================================================
dataset = LongitudinalDentaireDataset(
    csv_file=CSV_PATH,
    transform=clinical_vision_transforms["train"]
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=custom_collate_fn
)

# ==========================================================
# Nombre de classes
# ==========================================================
NUM_CLASSES = len(dataset.classes)

print(f"Nombre de classes détectées : {NUM_CLASSES}")

# ==========================================================
# Modèle
# ==========================================================
model = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(device)

# ==========================================================
# Loss + Optimizer
# ==========================================================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)

# ==========================================================
# Entraînement
# ==========================================================
EPOCHS = 10

print("\nDébut de l'entraînement...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for batch in loader:

        images = batch["images_pair"]

        labels = batch["label"]

        # moyenne Avant / Après
        images = images.mean(dim=1)

        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(loader)

    epoch_acc = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss = {epoch_loss:.4f} "
        f"Accuracy = {epoch_acc:.2f}%"
    )

# ==========================================================
# Sauvegarde
# ==========================================================
torch.save(model.state_dict(), MODEL_PATH)

print("\n==============================")
print("✅ Entraînement terminé")
print(f"✅ Modèle sauvegardé : {MODEL_PATH}")
print("==============================")