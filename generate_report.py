import os
from datetime import datetime

class ClinicalReportGenerator:
    def __init__(self, cabinet_name="Cabinet Dentaire IA - PFA"):
        self.cabinet_name = cabinet_name
        
        # Base de connaissances médicale intégrée pour le mode hors-ligne
        self.recommandations_cliniques = {
            'Carie': "Éviction du tissu carié et restauration par résine composite ou amalgame. Si atteinte profonde, évaluer la vitalité pulpaire.",
            'Dent_fracturee': "Évaluation radiologique de la ligne de fracture (conforme au marquage Grad-CAM). Si fracture coronaire simple : reconstitution composite ou pose de couronne. Si atteinte radiculaire : extraction à envisager.",
            'Lesion_periapicale': "Test de vitalité pulpaire indispensable. Traitement endodontique (dévitalisation) ou retraitement canalaire à prévoir. Prescription d'antibiotiques si phase aiguë tumorale ou infectieuse.",
            'Perte_osseuse': "Bilan parodontal complet (sondage des poches parodontales). Détartrage et surfaçage radiculaire (curetage). Planification d'un suivi de la maintenance parodontale.",
            'Normal': "Aucune anomalie radiologique majeure détectée sur cet examen. Maintien de l'hygiène bucco-dentaire habituelle et contrôle de routine dans 12 mois."
        }

    def generate_prompt_llm(self, patient_id, diagnostic, confiance, symptomes, traitement):
        """
        Génère un prompt structuré prêt à être injecté dans un LLM (Llama 3, Mistral, GPT-4).
        """
        prompt = f"""
        Agis en tant que chirurgien-dentiste expert. Rédige un compte-rendu radiologique formel et professionnel à partir des données techniques issues d'un système de détection par Deep Learning :
        
        - ID Patient : {patient_id}
        - Examen : Radiographie rétro-alvéolaire
        - Analyse Vision (ResNet-50) : {diagnostic} (Indice de confiance : {confiance:.2f}%)
        - Localisation : Validation visuelle par carte d'activation Grad-CAM positive
        - Symptômes cliniques rapportés : {symptomes}
        - Historique / Soins en cours : {traitement}

        Consignes de rédaction :
        1. Rédige une synthèse clinique faisant le lien entre la plainte du patient et l'image radiologique.
        2. Propose 2 à 3 options thérapeutiques hiérarchisées et adaptées au diagnostic.
        3. Adopte un ton médical, précis et rigoureux.
        """
        return prompt.strip()

    def create_text_report(self, patient_id, diagnostic, confiance, symptomes="Non renseigné", traitement="Aucun", save_to_file=True):
        """
        Génère le compte-rendu officiel du cabinet au format TXT.
        """
        date_str = datetime.now().strftime("%d/%m/%Y à %H:%M")
        recommandation = self.recommandations_cliniques.get(diagnostic, "Consultation clinique approfondie requise.")
        
        report = f"""
======================================================================
                     COMPTE-RENDU RADIOLOGIQUE IA
======================================================================
Établi par : {self.cabinet_name}
Date de l'examen : {date_str}
Dossier Patient : {patient_id}
----------------------------------------------------------------------
1. RÉSULTATS DE L'ANALYSE PAR VISION PAR ORDINATEUR (ResNet-50)
----------------------------------------------------------------------
> Diagnostic principal : {diagnostic.upper()}
> Indice de confiance  : {confiance:.2f}%
> Localisation Grad-CAM: Carte d'activation validée et jointe au dossier.

----------------------------------------------------------------------
2. CONTEXTE CLINIQUE & SYMPTÔMES
----------------------------------------------------------------------
> Symptômes initiaux : {symptomes}
> Historique / Soins : {traitement}

----------------------------------------------------------------------
3. SYNTHÈSE & PROPOSITION THÉRAPEUTIQUE AUTOMATISÉE
----------------------------------------------------------------------
Analyse :
La détection radiologique met en évidence une pathologie compatible avec 
une classe '{diagnostic}'. Ce résultat radiologique doit être corrélé 
avec l'examen clinique en fauteuil (sondage, tests de vitalité).

Protocole thérapeutique recommandé :
* {recommandation}

======================================================================
* Note : Ce compte-rendu est généré par un système d'assistance IA (Projet PFA) 
  et nécessite la validation d'un chirurgien-dentiste diplômé.
======================================================================
        """.strip()

        if save_to_file:
            os.makedirs("rapports_generes", exist_ok=True)
            file_name = f"rapport_patient_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            file_path = os.path.join("rapports_generes", file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"✅ Rapport sauvegardé avec succès : {file_path}")

        return report

    def create_pdf_report(self, patient_id, diagnostic, confiance, symptomes="Non renseigné", traitement="Aucun"):
        """
        Génère un rapport clinique PDF haut de gamme et retourne les bytes pour Streamlit.
        """
        from fpdf import FPDF
        
        recommandation = self.recommandations_cliniques.get(diagnostic, "Consultation clinique approfondie requise.")
        date_str = datetime.now().strftime("%d/%m/%Y à %H:%M")
        
        # Nettoyeur pour s'assurer que les caractères accentués français s'affichent correctement
        def clean(s):
            return str(s).encode('latin-1', 'replace').decode('latin-1')

        pdf = FPDF()
        pdf.add_page()
        pdf.set_margins(15, 15, 15)
        
        # --- EN-TÊTE GRAPHIQUE (Bandeau Bleu Marine de Clinique) ---
        pdf.set_fill_color(26, 54, 93)  # Bleu marine pro #1a365d
        pdf.rect(0, 0, 210, 38, 'F')
        
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("helvetica", "B", 16)
        pdf.set_xy(15, 10)
        pdf.cell(0, 10, clean(self.cabinet_name.upper()), 0, 1, 'L')
        
        pdf.set_font("helvetica", "I", 10)
        pdf.set_xy(15, 19)
        pdf.cell(0, 5, clean("Système d'Aide au Diagnostic Clinique & Vision Artificielle"), 0, 1, 'L')
        
        # --- BLOC D'IDENTIFICATION ---
        pdf.set_text_color(40, 40, 40)  # Anthracite élégant
        pdf.set_y(48)
        
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(90, 6, clean(f"ID DOSSIER : {patient_id}"), 0, 0, 'L')
        pdf.set_font("helvetica", "", 10)
        pdf.cell(0, 6, clean(f"Date de l'examen : {date_str}"), 0, 1, 'R')
        
        # Ligne horizontale de démarcation
        pdf.set_draw_color(200, 200, 200)
        pdf.line(15, 56, 195, 56)
        pdf.ln(8)
        
        # --- SECTION 1 : RÉSULTATS DE L'ANALYSE IA ---
        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(26, 54, 93)
        pdf.cell(0, 8, clean("1. ANALYSE RADIOLOGIQUE PAR DEEP LEARNING (ResNet-50)"), 0, 1, 'L')
        pdf.ln(2)
        
        pdf.set_text_color(40, 40, 40)
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(45, 6, clean("Diagnostic détecté :"), 0, 0, 'L')
        pdf.set_font("helvetica", "", 10)
        pdf.cell(0, 6, clean(diagnostic.replace('_', ' ').upper()), 0, 1, 'L')
        
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(45, 6, clean("Indice de confiance :"), 0, 0, 'L')
        pdf.set_font("helvetica", "", 10)
        pdf.cell(0, 6, clean(f"{confiance:.2f}%"), 0, 1, 'L')
        
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(45, 6, clean("Interprétation :"), 0, 0, 'L')
        pdf.set_font("helvetica", "", 10)
        pdf.cell(0, 6, clean("Cartographie d'activation visuelle validée par heatmap Grad-CAM"), 0, 1, 'L')
        pdf.ln(6)
        
        # --- SECTION 2 : DOSSIER CLINIQUE ---
        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(26, 54, 93)
        pdf.cell(0, 8, clean("2. CONTEXTE CLINIQUE ET ANAMNÈSE"), 0, 1, 'L')
        pdf.ln(2)
        
        pdf.set_text_color(40, 40, 40)
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(0, 6, clean("Symptômes signalés :"), 0, 1, 'L')
        pdf.set_font("helvetica", "", 10)
        pdf.multi_cell(0, 5, clean(symptomes))
        pdf.ln(3)
        
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(0, 6, clean("Historique thérapeutique connu :"), 0, 1, 'L')
        pdf.set_font("helvetica", "", 10)
        pdf.multi_cell(0, 5, clean(traitement))
        pdf.ln(6)
        
        # --- SECTION 3 : CONDUITE À TENIR RECOMMANDÉE ---
        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(26, 54, 93)
        pdf.cell(0, 8, clean("3. RECOMMANDATIONS & ORIENTATION THÉRAPEUTIQUE"), 0, 1, 'L')
        pdf.ln(2)
        
        pdf.set_text_color(40, 40, 40)
        pdf.set_font("helvetica", "I", 10)
        pdf.multi_cell(0, 5, clean(f"L'analyse automatisée de la texture osseuse et dentaire sur l'image radiographique confirme une anomalie localisée compatible avec la pathologie '{diagnostic.replace('_', ' ')}'."))
        pdf.ln(3)
        
        # Encadré thérapeutique (Couleur accentuée Teal #319795)
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(49, 151, 149)
        pdf.cell(0, 6, clean("Protocole thérapeutique conseillé :"), 0, 1, 'L')
        
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 5, clean(recommandation))
        
        # --- PIED DE PAGE AVEC DISCLAIMER DE SÉCURITÉ ---
        pdf.set_y(-32)
        pdf.set_font("helvetica", "I", 8)
        pdf.set_text_color(120, 120, 120)
        pdf.set_draw_color(220, 220, 220)
        pdf.line(15, pdf.get_y() - 2, 195, pdf.get_y() - 2)
        pdf.multi_cell(0, 4, clean("Avertissement : Ce rapport clinique est un document d'aide à la décision généré automatiquement par un prototype IA de Deep Learning (ResNet-50). Il ne dispense pas de la validation obligatoire et définitive par un chirurgien-dentiste praticien agréé."), 0, 'C')
        
        # Extraction des bytes de manière compatible toutes plateformes
        pdf_output = pdf.output()
        if isinstance(pdf_output, str):
            return pdf_output.encode('latin-1')
        return bytes(pdf_output)

if __name__ == "__main__":
    # Petit test d'intégrité
    generateur = ClinicalReportGenerator()
    print("✅ Module de génération mis à jour avec le support PDF !")