import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def find_optimal_thresholds(y_true, y_probs, class_names):
    best_thresholds = {}
    metrics_summary = {}

    for i, cls in enumerate(class_names):
        best_thresh = 0.50
        best_score = -1.0
        best_prec = 0.0
        best_rec = 0.0

        # Si aucune occurrence dans le dataset, fixer le seuil par défaut
        if np.sum(y_true[:, i]) == 0:
            best_thresholds[cls] = 0.50
            metrics_summary[cls] = {"seuil_optimal": 0.50, "precision": 0.0, "recall": 0.0}
            continue

        for thresh in np.arange(0.10, 0.85, 0.02):
            preds = (y_probs[:, i] >= thresh).astype(int)
            prec = precision_score(y_true[:, i], preds, zero_division=0)
            rec = recall_score(y_true[:, i], preds, zero_division=0)
            
            # Ponderation : Priorite absolue a la Precision (80% Precision, 20% Recall)
            score = 0.8 * prec + 0.2 * rec

            if score > best_score and (prec > 0 or rec > 0):
                best_score = score
                best_thresh = thresh
                best_prec = prec
                best_rec = rec

        best_thresholds[cls] = round(float(best_thresh), 3)
        metrics_summary[cls] = {
            "seuil_optimal": round(float(best_thresh), 3),
            "precision": round(float(best_prec * 100), 2),
            "recall": round(float(best_rec * 100), 2)
        }

    return best_thresholds, metrics_summary

def evaluate_with_thresholds(y_true, y_probs, thresholds_dict, class_names):
    y_preds = np.zeros_like(y_probs)
    for i, cls in enumerate(class_names):
        t = thresholds_dict.get(cls, 0.5)
        y_preds[:, i] = (y_probs[:, i] >= t).astype(int)

    report = classification_report(
        y_true, y_preds, target_names=class_names, zero_division=0, digits=4
    )
    return report, y_preds