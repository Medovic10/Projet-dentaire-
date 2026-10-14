import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config

def get_train_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.8),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

class DentalDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []

        patient_col = 'Dossier_Patient' if 'Dossier_Patient' in df.columns else 'patient_id'

        for _, row in self.df.iterrows():
            patient_name = row[patient_col]
            labels = row[config.CLASSES].values.astype(np.float32)
            patient_folder = os.path.join(self.img_dir, str(patient_name))

            if os.path.exists(patient_folder):
                for file in os.listdir(patient_folder):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(patient_folder, file), labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(labels)