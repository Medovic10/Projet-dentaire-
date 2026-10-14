import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. Charger le modèle et l'image
model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 5)
model.load_state_dict(torch.load("models/resnet50_dentaire_final.pth", map_location='cpu'))
model.eval()

# 2. Préparer l'image
img_path = "data/test_radio.jpeg" # ou .jpg
img = Image.open(img_path).convert('RGB')
preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
img_tensor = preprocess(img).unsqueeze(0)

# 3. Extraire les caractéristiques (Hook sur la dernière couche de convolution)
features = []
def hook(module, input, output): features.append(output)
model.layer4.register_forward_hook(hook)

# 4. Forward pass
output = model(img_tensor)
pred = torch.argmax(output)

# 5. Calcul de la Heatmap (Activation map)
activations = features[0].detach().squeeze()
weights = model.fc.weight[pred].detach() # Poids du diagnostic prédit
heatmap = torch.matmul(weights, activations.view(activations.size(0), -1)).view(activations.size(1), activations.size(2))

# 6. Visualisation
heatmap = F.relu(heatmap).numpy()
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) # Normalisation

plt.imshow(img.resize((224, 224)), alpha=0.5)
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.title(f"Zone d'intérêt pour : Diagnostic {pred.item()}")
plt.show()
# (Garde tout le début du script tel quel)

# 6. Superposition et Visualisation Professionnelle
# On redimensionne la heatmap pour qu'elle corresponde à la taille de l'image d'origine (ex: 1200x1200, pas 224x224)
heatmap_resized = np.array(Image.fromarray(heatmap).resize(img.size, Image.Resampling.LANCZOS))

# Création de la figure
plt.figure(figsize=(8, 8))

# Affichage de l'image originale en niveaux de gris (plus lisible pour une radio)
plt.imshow(img.convert('L'), cmap='gray')

# Affichage de la heatmap par-dessus, avec transparence (alpha) et une palette de couleurs (jet)
# On masque les valeurs très faibles de la heatmap pour ne garder que les zones d'intérêt principales
masked_heatmap = np.ma.masked_where(heatmap_resized < 0.5, heatmap_resized)
plt.imshow(masked_heatmap, cmap='jet', alpha=0.6, vmin=0, vmax=1)

plt.title(f"Carte d'activation pour : {CLASSES[pred.item()]}\nVisualisation de la zone de fracture")
plt.axis('off') # On enlève les axes pour un rendu plus propre

# Sauvegarde de l'image pour ton rapport
plt.savefig('resultat_gradcam_dent.png', bbox_inches='tight', dpi=300)
print(f"✅ Image de visualisation sauvegardée : resultat_gradcam_dent.png")

# Affichage de la fenêtre (si ton environnement le permet)
plt.show()