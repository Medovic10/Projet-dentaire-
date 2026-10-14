import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
CSV_PATH = r"D:\te\prj_dentaire\patients_preprocessed.csv"
DATASET_PATH = r"D:\te\PRJ\PRJ"
MODEL_SAVE_PATH = r"D:\te\prj_dentaire\best_dental_efficientnet.pth"

CLASSES = ["Normal", "Carie", "Fracture", "Lesion_periapicale", "Perte_osseuse", "Pulpite_aigue", "Maladie_parodontale"]
BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 5e-5  # Un taux plus faible évite un surapprentissage trop rapide
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"💻 Entraînement sur l'appareil : {DEVICE}")

# --- 1. DATASET PYTORCH ---
class DentalDataset(Dataset):
    def __init__(self, csv_path, img_dir, classes, patient_list, transform=None):
        self.df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')
        self.df = self.df[self.df['Dossier_Patient'].isin(patient_list)].reset_index(drop=True)
        self.img_dir = img_dir
        self.classes = classes
        self.transform = transform
        self.samples = []

        for _, row in self.df.iterrows():
            patient_name = row['Dossier_Patient']
            labels = row[self.classes].values.astype(np.float32)
            patient_folder = os.path.join(self.img_dir, str(patient_name))
            
            if os.path.exists(patient_folder):
                for file in os.listdir(patient_folder):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(patient_folder, file), labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(labels, dtype=torch.float32)

# --- 2. SÉPARATION DES DONNÉES (Fix de l'avertissement Arrow) ---
df_patients = pd.read_csv(CSV_PATH, sep=';', encoding='utf-8-sig')
# Correction : conversion explicite en liste Python standard
all_patients = list(df_patients['Dossier_Patient'].unique())

np.random.seed(42)
np.random.shuffle(all_patients)

split_idx = int(len(all_patients) * 0.8)
train_patients = all_patients[:split_idx]
val_patients = all_patients[split_idx:]

print(f"📊 Répartition : {len(train_patients)} patients pour l'entraînement, {len(val_patients)} pour la validation.")

# --- 3. TRANSFORMS (Data Augmentation renforcée) ---
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = DentalDataset(CSV_PATH, DATASET_PATH, CLASSES, train_patients, transform=train_transforms)
val_dataset = DentalDataset(CSV_PATH, DATASET_PATH, CLASSES, val_patients, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"📸 Images d'entraînement : {len(train_dataset)} | Images de validation : {len(val_dataset)}")

# --- 4. CONFIGURATION DU MODÈLE EFFICIENTNET-B0 ---
# Modèle plus léger et beaucoup plus robuste au surapprentissage
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Ajout d'un Dropout fort (0.5) pour casser la mémorisation
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_features, len(CLASSES))
)
model = model.to(DEVICE)

# Utilisation de AdamW avec de la décomposition des poids (L2 Regularization)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

# --- 5. BOUCLE D'ENTRAÎNEMENT ---
best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': [], 'val_f1_macro': []}

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        
    train_loss /= len(train_loader.dataset)
    
    # Validation
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            preds = torch.sigmoid(outputs).cpu().numpy()
            preds_bin = (preds > 0.5).astype(int)
            
            all_preds.extend(preds_bin)
            all_targets.extend(labels.cpu().numpy())
            
    val_loss /= len(val_loader.dataset)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_f1_macro'].append(f1)
    
    print(f"Epoch {epoch+1:02d}/{EPOCHS:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1-Score: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"🌟 Nouveau meilleur modèle sauvegardé ! (Loss: {val_loss:.4f})")

# --- 6. GRAPHIQUES ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Courbe de Perte (Loss) - EfficientNet-B0')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['val_f1_macro'], label='Val F1 (Macro)', color='green')
plt.title('Évolution du F1-Score - EfficientNet-B0')
plt.xlabel('Epochs')
plt.ylabel('F1-Score')
plt.legend()

plt.tight_layout()
plt.savefig(r"D:\te\prj_dentaire\performance_curves.png")
print("📈 Courbes de performances mises à jour sauvegardées !")