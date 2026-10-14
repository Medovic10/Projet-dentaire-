import sqlite3
import io
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

DB_NAME = "patients.db"

def init_db():
    """Initialise la table des patients dans SQLite si elle n'existe pas."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            name TEXT,
            age INTEGER,
            date TEXT,
            pathology TEXT,
            confidence TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_patient(patient_id, name, age, pathology, confidence):
    """Enregistre ou met à jour les informations d'un patient."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT OR REPLACE INTO patients (id, name, age, date, pathology, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (patient_id, name, age, date_str, pathology, confidence))
    conn.commit()
    conn.close()

def get_all_patients():
    """Récupère l'historique de tous les patients enregistrés."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, age, date, pathology, confidence FROM patients ORDER BY date DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def generate_pdf_bytes(patient_id, name, age, pathology, confidence):
    """Génère un rapport de diagnostic PDF médical hautement professionnel."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Styles typographiques
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=12
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=10,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#334155'),
        leading=14
    )
    
    # 1. En-tête
    story.append(Paragraph("🦷 DENTAL AI STUDIO", title_style))
    story.append(Paragraph("Plateforme d'Aide au Diagnostic Radiologique & Imagerie Dentaire", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=15))
    
    # 2. Informations Examen & Patient
    date_now = datetime.datetime.now().strftime('%d/%m/%Y à %H:%M')
    
    info_data = [
        [
            Paragraph("<b>ID Examen :</b>", body_style), Paragraph(str(patient_id), body_style),
            Paragraph("<b>Date d'examen :</b>", body_style), Paragraph(date_now, body_style)
        ],
        [
            Paragraph("<b>Patient :</b>", body_style), Paragraph(str(name), body_style),
            Paragraph("<b>Âge :</b>", body_style), Paragraph(f"{age} ans" if str(age).isdigit() else str(age), body_style)
        ],
        [
            Paragraph("<b>Modèle IA :</b>", body_style), Paragraph("EfficientNet Ensemble Multi-Label", body_style),
            Paragraph("<b>Statut :</b>", body_style), Paragraph("<font color='#D97706'><b>Pré-interprété par IA</b></font>", body_style)
        ]
    ]
    
    info_table = Table(info_data, colWidths=[1.3*inch, 2.2*inch, 1.3*inch, 2.2*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#F1F5F9')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 15))
    
    # 3. Synthèse Diagnostic
    story.append(Paragraph("🔬 Synthèse des Détections Radiologiques", section_heading))
    
    # Formatage propre de la chaîne des pathologies (ex: "Pulpite_aigue" -> "Pulpite aigue")
    clean_patho = pathology.replace('_', ' ')
    
    if "Sain" in clean_patho or "R.A.S" in clean_patho:
        status_color = "#166534"
        status_bg = "#DCFCE7"
        badge_text = "AUCUNE ANOMALIE MAJEURE"
    else:
        status_color = "#991B1B"
        status_bg = "#FEE2E2"
        badge_text = "ANOMALIE(S) DÉTECTÉE(S)"

    diag_data = [
        [
            Paragraph("<b>Pathologie(s) Identifiée(s)</b>", body_style),
            Paragraph("<b>Confiance IA</b>", body_style),
            Paragraph("<b>Évaluation</b>", body_style)
        ],
        [
            Paragraph(f"<b><font color='{status_color}'>{clean_patho}</font></b>", body_style),
            Paragraph(f"<b>{confidence}</b>", body_style),
            Paragraph(f"<font color='{status_color}'><b>{badge_text}</b></font>", body_style)
        ]
    ]
    
    diag_table = Table(diag_data, colWidths=[3.2*inch, 1.6*inch, 2.2*inch])
    diag_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor(status_bg)),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (1,0), (1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 15))
    
    # 4. Recommandations
    story.append(Paragraph("📌 Recommandations Cliniques", section_heading))
    notes_text = """
    • Ce rapport est issu d'un algorithme de vision par ordinateur spécialisé en imagerie panoramique et rétro-alvéolaire.<br/>
    • Il est recommandé d'analyser la carte de chaleur d'explicabilité (Grad-CAM) sur la station de travail.<br/>
    • La décision thérapeutique finale reste sous la responsabilité exclusive du praticien.
    """
    story.append(Paragraph(notes_text, body_style))
    story.append(Spacer(1, 20))
    
    # 5. Bloc de validation / Signature
    story.append(KeepTogether([
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceAfter=15),
        Table([
            [
                Paragraph("<b>Validation du Chirurgien-Dentiste :</b>", body_style),
                Paragraph("<b>Date & Signature :</b>", body_style)
            ],
            [
                Paragraph("<br/><br/>Dr. ____________________________________", body_style),
                Paragraph("<br/><br/>____________________________________", body_style)
            ]
        ], colWidths=[4.2*inch, 2.8*inch], style=[
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('PADDING', (0,0), (-1,-1), 0)
        ])
    ]))
    
    story.append(Spacer(1, 20))
    
    # 6. Mentions légales en bas de page
    footer_style = ParagraphStyle(
        'FooterText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        textColor=colors.HexColor('#475569'),
        alignment=1
    )
    legal_text = "AVERTISSEMENT : Document d'assistance médicale généré par Dental AI Studio. Ce rapport ne constitue pas un acte de diagnostic autonome et nécessite la validation d'un professionnel de santé diplômé."
    
    footer_table = Table([[Paragraph(legal_text, footer_style)]], colWidths=[7.0*inch])
    footer_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F1F5F9')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(footer_table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()