import os
import random
import pandas as pd

# Configuration des chemins
CSV_INPUT = "metadata/dataset_initial.csv"
CSV_OUTPUT = "metadata/dataset_test_fictif.csv"

# Base de données fictive : Triplets cliniquement cohérents
TRIPLETS_CLINIQUES = [
    (
        "Douleur aiguë au contact du froid et des aliments sucrés.",
        "Carie",
        "Obturation en composite (Plombage esthétique)",
    ),
    (
        "Sensibilité prolongée au chaud et douleur irradiante vers la mâchoire.",
        "Carie",
        "Nettoyage de la cavité et protection pulpaire",
    ),
    (
        "Douleur pulsatile nocturne, sensation de dent longue et léger gonflement.",
        "Lesion_periapicale",
        "Traitement endodontique (Dévitalisation et désinfection du canal)",
    ),
    (
        "Abcès localisé avec présence d'une fistule et douleur à la pression.",
        "Lesion_periapicale",
        "Drainage de l'abcès et traitement antibiotique avant reprise du canal",
    ),
    (
        "Gêne à la mastication sur une dent ayant un ancien plombage profond.",
        "Traitement_canal",
        "Retraitement canalaire et pose d'un inlay-core",
    ),
    (
        "Fragilité de la paroi dentaire constatée après un traitement de racine.",
        "Traitement_canal",
        "Pose d'une couronne en céramique de protection",
    ),
    (
        "Douleur vive et soudaine lors de la morsure d'un aliment dur.",
        "Dent_fracturee",
        "Extraction de la racine fracturée et planification d'un implant",
    ),
    (
        "Fêlure visible sur l'émail avec sensibilité ponctuelle au brossage.",
        "Dent_fracturee",
        "Reconstitution en résine composite haute densité",
    ),
    (
        "Mobilité dentaire légère, récession gingivale et saignements fréquents.",
        "Perte_osseuse",
        "Surfaçage radiculaire et assainissement parodontal",
    ),
    (
        "Tassement alimentaire fréquent entre les dents et inflammation chronique.",
        "Perte_osseuse",
        "Comblement osseux biomatériau et contentions",
    ),
    (
        "Aucun symptôme rapporté par le patient. Visite de contrôle annuelle.",
        "Normal",
        "Aucun traitement requis. Détartrage et polissage préventifs",
    ),
]


def generate_full_mock_data():
    if not os.path.exists(CSV_INPUT):
        print(
            f"❌ Erreur : Le fichier {CSV_INPUT} est introuvable. Lance d'abord l'extraction PDF !"
        )
        return

    # Lecture en essayant de détecter le bon format
    df = pd.read_csv(CSV_INPUT, sep=";")
    print(f"--- 💉 Injection de données cliniques fictives pour {len(df)} patients ---")

    symptomes_list = []
    diagnostics_list = []
    traitements_list = []

    for i in range(len(df)):
        id_pat = df.iloc[i]["id_patient"]
        symptome, diagnostic, traitement = random.choice(TRIPLETS_CLINIQUES)

        symptomes_list.append(symptome)
        diagnostics_list.append(diagnostic)
        traitements_list.append(traitement)

        print(
            f"✔️ Pat [{id_pat}] -> Diag: {diagnostic.ljust(18)} | Traitement: {traitement[:30]}..."
        )

    df["symptomes_initiaux"] = symptomes_list
    df["diagnostic"] = diagnostics_list
    df["traitement_realise"] = traitements_list

    # ----------------------------------------------------------------------------------
    # CORRECTION ICI : 'utf-8-sig' force Excel à lire parfaitement les accents français
    # ----------------------------------------------------------------------------------
    df.to_csv(CSV_OUTPUT, index=False, sep=";", encoding="utf-8-sig")

    print("\n========================================================")
    print("✅ SUCCÈS ! Fichier de test généré avec le bon encodage Excel :")
    print(f"📁 {CSV_OUTPUT}")
    print("========================================================")


if __name__ == "__main__":
    generate_full_mock_data()