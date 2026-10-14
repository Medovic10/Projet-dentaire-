import pandas as pd
import os
import random

CSV_PATH = r"D:\te\prj_dentaire\patients_diagnostics.csv"

# Dictionnaire de templates de phrases médicales pour varier les notes
OBSERVATIONS = {
    "Normal": [
        "Examen clinique normal. Absence d'anomalie visible ou de lésion active. Tissus parodontaux sains.",
        "Intégrité coronaire et radiculaire conservée. Pas de foyer infectieux visible.",
        "Dentition saine, structures d'appui osseuses physiologiques et stables."
    ],
    "Carie": [
        "Mise en évidence d'une perte de substance coronaire radiotransparente évocatrice d'une lésion carieuse active.",
        "Lésion carieuse profonde suspectée au niveau amélo-dentinaire.",
        "Zone d'ombre radiologique coronaire confirmant une reprise de carie sous-jacente."
    ],
    "Fracture": [
        "Mise en évidence d'un trait de fracture coronaire net avec atteinte possible de la chambre pulpaire.",
        "Fêlure/fracture visible sur la face mésio-distale, intégrité radiculaire à surveiller.",
        "Fracture coronaire partielle suite à un traumatisme, limites cervicales saines."
    ],
    "Lesion_periapicale": [
        "Image radioclaire péri-apicale bien délimitée traduisant une ostéite péri-radiculaire d'origine endodontique.",
        "Lésion d'origine endodontique (LIPOE) visible à l'apex de la racine.",
        "Raréfaction osseuse péri-apicale marquée nécessitant un retraitement canalaire."
    ],
    "Perte_osseuse": [
        "Résorption osseuse alvéolaire de type horizontal modérée.",
        "Perte de hauteur de la crête osseuse alvéolaire avec élargissement de l'espace desmodontal.",
        "Alvéolyse verticale localisée avec atteinte de la furcation suspectée."
    ],
    "Pulpite_aigue": [
        "Symptomatologie compatible avec une inflammation pulpaire irréversible (pulpite aiguë). Traitement d'urgence requis.",
        "Atteinte pulpaire profonde et douloureuse nécessitant une biopulpectomie.",
        "Hyperémie pulpaire sévère secondaire à une complication carieuse."
    ],
    "Maladie_parodontale": [
        "Signes de parodontite active avec perte d'attache et lyse osseuse généralisée.",
        "Atteinte globale des tissus de soutien de la dent avec présence de dépôts tartriques sous-gingivaux radiopaques.",
        "Poche parodontale profonde associée à une résorption osseuse alvéolaire diffuse."
    ]
}

if not os.path.exists(CSV_PATH):
    print(f"Erreur : Le fichier {CSV_PATH} n'existe pas.")
    exit()

# Chargement du fichier
df = pd.read_csv(CSV_PATH, sep=';', encoding='utf-8-sig', keep_default_na=False)

print("⚡ Déduction et génération des notes cliniques en cours...")

for idx, row in df.iterrows():
    diagnostics_str = str(row['Diagnostic_Principal']).strip()
    
    # Si aucun diagnostic n'est renseigné, on passe par défaut à "Normal"
    if not diagnostics_str:
        diagnostics = ["Normal"]
    else:
        diagnostics = [d.strip() for d in diagnostics_str.split(",") if d.strip()]

    notes_generées = []
    
    # On pioche une phrase médicale aléatoire pour chaque diagnostic présent
    for diag in diagnostics:
        if diag in OBSERVATIONS:
            phrase = random.choice(OBSERVATIONS[diag])
            notes_generées.append(f"- {diag} : {phrase}")
        else:
            notes_generées.append(f"- {diag} : Anomalie détectée à l'examen.")

    # On assemble le tout dans la colonne Notes_Cliniques
    df.at[idx, 'Notes_Cliniques'] = "\n".join(notes_generées)

# Sauvegarde finale
df.to_csv(CSV_PATH, sep=';', index=False, encoding='utf-8-sig')
print("✅ Génération terminée ! Le fichier CSV a été enrichi avec succès.")