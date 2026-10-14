import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# =====================================================================
# 1. PIPELINE D'AUGMENTATION CLINIQUE SÉCURISÉ
# =====================================================================
# Suppression du miroir horizontal pour respecter la nomenclature FDI
clinical_vision_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

# =====================================================================
# 2. CLASSE DATALOADER LONGITUDINAL MULTIMODAL
# =====================================================================
class LongitudinalDentaireDataset(Dataset):
    """
    Dataset PyTorch multimodal conçu pour ingérer l'historique longitudinal 
    des patients (Imagerie médicale + Textes cliniques).
    """
    def __init__(self, csv_file, root_dir="", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Lecture du CSV sécurisée en UTF-8 avec gestion du BOM
        self.data_frame = pd.read_csv(csv_file, sep=';', encoding='utf-8-sig')
        self.data_frame = self.data_frame.dropna(subset=['diagnostic'])
        
        # Encodage automatique des labels catégoriels en indices entiers
        self.classes = sorted(list(set(self.data_frame['diagnostic'].unique())))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"--- [Dataset V2] Initialisation réussie ---")
        print(f"Patients valides chargés : {len(self.data_frame)}")
        print(f"Mapping des diagnostics : {self.class_to_idx}")

    def __len__(self):
        return len(self.data_frame)

    def _load_and_transform_image(self, rel_path):
        """Helper interne pour charger et transformer une image de manière sécurisée."""
        if pd.isna(rel_path) or not str(rel_path).strip():
            # Retourne un tenseur noir par défaut si l'image manque (Padding spatial)
            return torch.zeros((3, 224, 224), dtype=torch.float32)
            
        full_path = os.path.join(self.root_dir, str(rel_path).strip())
        if not os.path.exists(full_path):
            print(f"⚠️ Avertissement : Fichier introuvable -> {full_path}")
            return torch.zeros((3, 224, 224), dtype=torch.float32)
            
        image = Image.open(full_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        
        # 1. Traitement de la branche Vision (Paire canonique Avant / Après)
        img_avant = self._load_and_transform_image(row['image_avant_path'])
        
        # Si l'image après n'existe pas, on duplique l'image avant pour maintenir la dimension du batch
        path_apres = row['image_apres_path']
        if pd.isna(path_apres) or not str(path_apres).strip():
            img_apres = img_avant.clone()
        else:
            img_apres = self._load_and_transform_image(path_apres)

        # Empilement sur une nouvelle dimension : Tenseur de forme [2, 3, 224, 224]
        images_longitudinales = torch.stack([img_avant, img_apres], dim=0)

        # 2. Extraction des chemins intermédiaires
        inter_str = str(row['images_intermediaires']) if pd.notna(row['images_intermediaires']) else ""
        inter_paths = [p.strip() for p in inter_str.split('|') if p.strip()] if inter_str else []

        # 3. Traitement des labels et métadonnées cliniques (Branche NLP)
        label_str = row['diagnostic']
        label_idx = self.class_to_idx[label_str]

        # Encapsulation dans un dictionnaire structuré pour l'entraînement mutimodal
        sample = {
            'id_patient': str(row['id_patient']),
            'images_pair': images_longitudinales,               # Tenseur [2, 3, 224, 224]
            'label': torch.tensor(label_idx, dtype=torch.long), # Tenseur scalaire
            'num_intermediaires': len(inter_paths),             # Métadonnée entière
            'intermediaires_paths': inter_paths,                # Liste de chaînes
            'symptomes': str(row['symptomes_initiaux']),        # Texte brut pour LLM
            'traitement': str(row['traitement_realise'])        # Texte brut pour LLM
        }

        return sample
    
# =====================================================================
# 4. CUSTOM COLLATE FUNCTION
# =====================================================================

def custom_collate_fn(batch):
    """
    Fonction personnalisée pour empiler correctement un batch multimodal.
    """

    images_pair = torch.stack([item["images_pair"] for item in batch])

    labels = torch.stack([item["label"] for item in batch])

    ids = [item["id_patient"] for item in batch]

    symptomes = [item["symptomes"] for item in batch]

    traitements = [item["traitement"] for item in batch]

    inter_paths = [item["intermediaires_paths"] for item in batch]

    num_inter = torch.tensor(
        [item["num_intermediaires"] for item in batch],
        dtype=torch.long
    )

    return {
        "id_patient": ids,
        "images_pair": images_pair,
        "label": labels,
        "symptomes": symptomes,
        "traitement": traitements,
        "intermediaires_paths": inter_paths,
        "num_intermediaires": num_inter
    }
# =====================================================================
# 3. SCRIPT DE VÉRIFICATION ET VALIDATION TECHNIQUE
# =====================================================================
if __name__ == "__main__":
    csv_test_path = "metadata/dataset_test_fictif.csv"
    
    if os.path.exists(csv_test_path):
        # On utilise directement nos vraies transformations cliniques !
        dataset = LongitudinalDentaireDataset(
            csv_file=csv_test_path, 
            transform=clinical_vision_transforms['train']
        )
        
        # Test avec un DataLoader en batch de 2
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        batch = next(iter(loader))
        
        print("\n--- ✅ Test du Batch Loader réussi ---")
        print(f"Forme du tenseur d'images (Batch, Paire, Canaux, H, W) : {batch['images_pair'].shape}")
        print(f"Labels générés : {batch['label']}")
        print(f"Exemple de symptômes extraits : {batch['symptomes'][0][:50]}...")
        print(f"Compte d'images intermédiaires dans le batch : {batch['num_intermediaires']}")
    else:
        print(f"❌ Erreur : Fichier {csv_test_path} introuvable.")