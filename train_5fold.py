import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

import config
from src.dataset import DentalDataset, get_train_transforms, get_val_transforms
from src.model import get_efficientnet_model
from src.utils import find_optimal_thresholds, evaluate_with_thresholds

def train_one_fold(fold, train_loader, val_loader, model, criterion, optimizer, scheduler):
    best_val_loss = float('inf')
    best_model_path = os.path.join(config.MODEL_DIR, f"efficientnet_fold_{fold}.pth")

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(outputs)
                val_preds.append(probs.cpu().numpy())
                val_targets.append(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        print(f"Fold {fold} | Époque {epoch+1:02d}/{config.EPOCHS:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    fold_probs = []
    fold_true = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(config.DEVICE)
            probs = torch.sigmoid(model(images)).cpu().numpy()
            fold_probs.append(probs)
            fold_true.append(labels.numpy())

    return np.vstack(fold_probs), np.vstack(fold_true)

def main():
    print(f"🚀 Lancement de la 5-Fold Cross Validation sur : {config.DEVICE.upper()}")

    if config.METADATA_PATH.endswith('.xlsx'):
        df = pd.read_excel(config.METADATA_PATH)
    else:
        try:
            df = pd.read_csv(config.METADATA_PATH, sep=';', encoding='utf-8-sig')
            if 'Dossier_Patient' not in df.columns and 'patient_id' not in df.columns:
                df = pd.read_csv(config.METADATA_PATH, sep=',')
        except Exception:
            df = pd.read_csv(config.METADATA_PATH)

    patient_col = 'Dossier_Patient' if 'Dossier_Patient' in df.columns else 'patient_id'
    groups = df[patient_col].values

    gkf = GroupKFold(n_splits=config.NUM_FOLDS)
    
    oof_predictions_list = []
    oof_targets_list = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups), 1):
        print(f"\n==================== FOLD {fold}/{config.NUM_FOLDS} ====================")
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_dataset = DentalDataset(train_df, config.DATA_DIR, transform=get_train_transforms(config.IMAGE_SIZE))
        val_dataset = DentalDataset(val_df, config.DATA_DIR, transform=get_val_transforms(config.IMAGE_SIZE))

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

        model = get_efficientnet_model(num_classes=config.NUM_CLASSES, dropout_prob=config.DROPOUT).to(config.DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        val_probs, val_true = train_one_fold(fold, train_loader, val_loader, model, criterion, optimizer, scheduler)

        oof_predictions_list.append(val_probs)
        oof_targets_list.append(val_true)

    oof_predictions = np.vstack(oof_predictions_list)
    oof_targets = np.vstack(oof_targets_list)

    print("\n==================================================")
    print("🎯 Optimisation des Seuils Dynamiques par Pathologie")
    print("==================================================")
    optimal_thresholds, metrics_summary = find_optimal_thresholds(oof_targets, oof_predictions, config.CLASSES)

    thresholds_file = os.path.join(config.MODEL_DIR, "optimal_thresholds.json")
    with open(thresholds_file, "w", encoding="utf-8") as f:
        json.dump(optimal_thresholds, f, indent=4, ensure_ascii=False)

    print(f"✅ Seuils sauvegardés dans : {thresholds_file}")
    for cls, metrics in metrics_summary.items():
        print(f" • {cls:<20} | Seuil: {metrics['seuil_optimal']} | Précision: {metrics['precision']}% | Rappel: {metrics['recall']}%")

    report, _ = evaluate_with_thresholds(oof_targets, oof_predictions, optimal_thresholds, config.CLASSES)
    print("\n📊 RAPPORT CLINIQUE FINAL D'ÉVALUATION (OOF 5-FOLD) :")
    print(report)

if __name__ == "__main__":
    main()