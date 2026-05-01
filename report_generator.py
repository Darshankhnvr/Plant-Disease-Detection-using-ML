import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_case_report(case_data, treatments, yield_predictions):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='SubHeading', parent=styles['Heading2'], fontSize=14, spaceAfter=10))
    styles.add(ParagraphStyle(name='NormalText', parent=styles['Normal'], fontSize=11, spaceAfter=8))
    
    Story = []
    
    # Title
    title = f"CropSense Disease Report: {case_data['case']['plant_name']}"
    Story.append(Paragraph(title, styles['Heading1']))
    Story.append(Spacer(1, 12))
    
    # Case Overview
    Story.append(Paragraph("Case Overview", styles['SubHeading']))
    case_info = [
        ["Case ID:", str(case_data['case']['id'])],
        ["Created Date:", case_data['case']['created_date']],
        ["Status:", case_data['case']['status'].title()],
        ["Initial Disease:", case_data['case']['initial_disease'].replace('___', ' - ').replace('_', ' ')],
        ["Initial Severity:", case_data['case']['initial_severity']],
        ["Initial Health Score:", f"{case_data['case']['initial_health_score']}/100"]
    ]
    
    t = Table(case_info, colWidths=[120, 300])
    t.setStyle(TableStyle([
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    Story.append(t)
    Story.append(Spacer(1, 20))
    
    # Progression History
    Story.append(Paragraph("Progression History", styles['SubHeading']))
    if case_data['progression']:
        prog_data = [["Date", "Disease", "Severity", "Health Score"]]
        for p in case_data['progression']:
            prog_data.append([
                p['analysis_date'][:10],
                p['disease_detected'].replace('___', ' - ').replace('_', ' '),
                f"{p['confidence']}% Conf.",
                f"{p['health_score']}/100"
            ])
            
        t_prog = Table(prog_data, colWidths=[80, 180, 80, 80])
        t_prog.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e7d32')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0'))
        ]))
        Story.append(t_prog)
    else:
        Story.append(Paragraph("No progression history recorded.", styles['NormalText']))
    Story.append(Spacer(1, 20))
    
    # Treatments
    Story.append(Paragraph("Treatments Applied", styles['SubHeading']))
    if treatments:
        treat_data = [["Date", "Treatment", "Dosage", "Cost"]]
        for t in treatments:
            treat_data.append([
                t[3][:10], # date
                t[2],      # name
                t[4] if t[4] else "N/A", # dosage
                f"${t[6]}" if t[6] else "N/A" # cost
            ])
            
        t_treat = Table(treat_data, colWidths=[80, 180, 100, 60])
        t_treat.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff9800')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fff3e0')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#ffcc80'))
        ]))
        Story.append(t_treat)
    else:
        Story.append(Paragraph("No treatments recorded.", styles['NormalText']))
    Story.append(Spacer(1, 20))
    
    # Yield Predictions
    Story.append(Paragraph("Economic Impact Analysis", styles['SubHeading']))
    if yield_predictions:
        yp = yield_predictions[-1] # latest prediction
        yp_info = [
            ["Expected Yield Loss:", f"{yp[2]}%"],
            ["Economic Impact:", f"${yp[3]}"],
            ["Prediction Date:", yp[4][:10]]
        ]
        t_yp = Table(yp_info, colWidths=[150, 270])
        t_yp.setStyle(TableStyle([
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        Story.append(t_yp)
    else:
        Story.append(Paragraph("No economic impact predictions available.", styles['NormalText']))
        
    Story.append(Spacer(1, 40))
    Story.append(Paragraph(f"Generated by CropSense on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Italic']))

    doc.build(Story)
    buffer.seek(0)
    return buffer
