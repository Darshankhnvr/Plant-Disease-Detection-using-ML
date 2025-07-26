from flask import Flask, render_template,request,redirect,send_from_directory,url_for,jsonify
import numpy as np
import json
import uuid
import tensorflow as tf
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
import os
from PIL import Image as PILImage
import io
from database import DiseaseTracker
from yield_predictor import YieldPredictor

app = Flask(__name__)
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")

# Initialize database and yield predictor
db_tracker = DiseaseTracker()
yield_predictor = YieldPredictor()

# Custom Jinja2 filter for JSON parsing
@app.template_filter('fromjson')
def fromjson_filter(value):
    """Parse JSON string to Python object"""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
label = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

with open("plant_disease.json",'r') as file:
    plant_disease = json.load(file)

# print(plant_disease[4])

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/',methods = ['GET'])
def home():
    return render_template('home.html')

def extract_features(image):
    image = tf.keras.utils.load_img(image,target_size=(160,160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature



def calculate_confidence_score(prediction):
    """Calculate confidence percentage"""
    return float(prediction.max() * 100)

def assess_disease_severity(disease_name, confidence):
    """Assess disease severity based on confidence and disease type"""
    if "healthy" in disease_name.lower():
        return "Healthy", "#4caf50"
    
    # Severity thresholds based on confidence
    if confidence >= 90:
        return "Severe", "#f44336"
    elif confidence >= 70:
        return "Moderate", "#ff9800"
    else:
        return "Mild", "#ffeb3b"

def calculate_health_score(disease_name, confidence, severity):
    """Calculate overall plant health score (0-100)"""
    base_score = 100
    
    if "healthy" in disease_name.lower():
        return min(100, base_score - (100 - confidence))
    
    # Disease penalty based on type
    disease_penalties = {
        "blight": 40,
        "rot": 35,
        "rust": 25,
        "spot": 20,
        "mildew": 15,
        "virus": 45,
        "bacterial": 30
    }
    
    # Find applicable penalty
    penalty = 25  # default penalty
    for disease_type, disease_penalty in disease_penalties.items():
        if disease_type in disease_name.lower():
            penalty = disease_penalty
            break
    
    # Severity multiplier
    severity_multipliers = {
        "Mild": 0.5,
        "Moderate": 0.8,
        "Severe": 1.2
    }
    
    final_penalty = penalty * severity_multipliers.get(severity, 0.8)
    health_score = max(0, base_score - final_penalty)
    
    return round(health_score)



def get_top_predictions(prediction, top_n=3):
    """Get top N predictions for reference only"""
    top_predictions = []
    
    # Get indices of top predictions
    top_indices = np.argsort(prediction[0])[::-1][:top_n]
    
    for idx in top_indices:
        confidence = float(prediction[0][idx] * 100)
        # Only include if confidence is meaningful (>5%)
        if confidence > 5.0:
            disease_info = {
                'name': label[idx].replace('___', ' - ').replace('_', ' '),
                'confidence': confidence,
                'details': plant_disease[idx]
            }
            top_predictions.append(disease_info)
    
    return top_predictions

def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    
    # Primary prediction
    primary_idx = prediction.argmax()
    primary_disease = plant_disease[primary_idx]
    confidence = calculate_confidence_score(prediction)
    
    # Disease severity assessment
    severity, severity_color = assess_disease_severity(primary_disease['name'], confidence)
    
    # Health score calculation
    health_score = calculate_health_score(primary_disease['name'], confidence, severity)
    
    # Get top predictions for reference (optional - can be used in UI)
    top_predictions = get_top_predictions(prediction, top_n=3)
    
    # Compile result - focusing on primary prediction only
    result = {
        'primary_disease': primary_disease,
        'confidence': round(confidence, 1),
        'severity': severity,
        'severity_color': severity_color,
        'health_score': health_score,
        'top_predictions': top_predictions,  # Optional: for showing alternative possibilities
        'has_multiple': False  # Set to False since we're only showing primary result
    }
    
    return result

def generate_pdf_report(prediction_data, image_path, output_path):
    """Generate a comprehensive PDF report of the plant disease analysis"""
    
    try:
        # Create the PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=0.5*inch)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2e7d32')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#1b5e20')
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            alignment=TA_JUSTIFY
        )
        
        # Title
        story.append(Paragraph("CropSense - Plant Disease Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Report metadata
        report_info = [
            ['Report Generated:', datetime.now().strftime("%B %d, %Y at %I:%M %p")],
            ['Analysis System:', 'CropSense AI v1.0'],
            ['Model Accuracy:', '99%+'],
            ['Processing Time:', '< 5 seconds']
        ]
        
        info_table = Table(report_info, colWidths=[2*inch, 3*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f5e8')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2e7d32')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#c8e6c9')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
        ]))
        
        story.append(info_table)
        story.append(Spacer(1, 30))
        
        # Add image if it exists
        if os.path.exists(image_path):
            try:
                # Resize image for PDF
                with PILImage.open(image_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize to fit in PDF
                    img.thumbnail((300, 300), PILImage.Resampling.LANCZOS)
                    
                    # Save to temporary buffer
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='JPEG', quality=85)
                    img_buffer.seek(0)
                    
                    # Create reportlab Image
                    pdf_image = Image(img_buffer, width=3*inch, height=3*inch)
                    
                    # Center the image
                    story.append(Paragraph("Analyzed Plant Image", heading_style))
                    story.append(Spacer(1, 10))
                    
                    # Create a table to center the image
                    image_table = Table([[pdf_image]], colWidths=[6*inch])
                    image_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))
                    story.append(image_table)
                    story.append(Spacer(1, 20))
                    
            except Exception as e:
                story.append(Paragraph(f"Image could not be processed: {str(e)}", normal_style))
                story.append(Spacer(1, 20))
        
        # Primary Diagnosis Section
        story.append(Paragraph("Primary Diagnosis", heading_style))
        
        primary_disease = prediction_data['primary_disease']
        disease_name = primary_disease['name'].replace('___', ' - ').replace('_', ' ')
        
        diagnosis_data = [
            ['Disease Detected:', disease_name],
            ['Confidence Level:', f"{prediction_data['confidence']}%"],
            ['Severity Assessment:', prediction_data['severity']],
            ['Plant Health Score:', f"{prediction_data['health_score']}/100"]
        ]
        
        diagnosis_table = Table(diagnosis_data, colWidths=[2*inch, 3*inch])
        diagnosis_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ffebee')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#d32f2f')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#ffcdd2')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#fafafa')])
        ]))
        
        story.append(diagnosis_table)
        story.append(Spacer(1, 20))
        
        # Disease Information
        story.append(Paragraph("Disease Information", heading_style))
        
        story.append(Paragraph("<b>Cause:</b>", normal_style))
        story.append(Paragraph(primary_disease['cause'], normal_style))
        story.append(Spacer(1, 10))
        
        story.append(Paragraph("<b>Treatment Recommendations:</b>", normal_style))
        story.append(Paragraph(primary_disease['cure'], normal_style))
        story.append(Spacer(1, 20))
        
        # Alternative Possibilities (if any)
        if len(prediction_data['top_predictions']) > 1:
            story.append(Paragraph("Alternative Possibilities", heading_style))
            
            alt_data = [['Disease Name', 'Confidence', 'Notes']]
            for pred in prediction_data['top_predictions'][1:3]:  # Skip first (primary) and show next 2
                alt_data.append([
                    pred['name'],
                    f"{pred['confidence']:.1f}%",
                    "Lower confidence alternative"
                ])
            
            alt_table = Table(alt_data, colWidths=[2.5*inch, 1*inch, 2*inch])
            alt_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f5e8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2e7d32')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#c8e6c9')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
            ]))
            
            story.append(alt_table)
            story.append(Spacer(1, 20))
        
        # Recommendations Section
        story.append(Paragraph("General Recommendations", heading_style))
        
        recommendations = [
            "• Monitor the affected plant regularly for changes in symptoms",
            "• Follow the specific treatment recommendations provided above",
            "• Ensure proper plant nutrition and watering practices",
            "• Remove and dispose of severely affected plant parts properly",
            "• Consider consulting with a local agricultural extension office for severe cases",
            "• Take preventive measures to protect other plants in the area"
        ]
        
        for rec in recommendations:
            story.append(Paragraph(rec, normal_style))
        
        story.append(Spacer(1, 30))
        
        # Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#666666')
        )
        
        story.append(Paragraph("This report was generated by CropSense AI Plant Disease Recognition System", footer_style))
        story.append(Paragraph("For more information, visit our website or contact support", footer_style))
        
        # Build the PDF
        doc.build(story)
        return True
        
    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        return False

@app.route('/download-report')
def download_report():
    """Generate and download PDF report"""
    try:
        # Get the prediction data from session or request
        # For now, we'll need to pass this data via URL parameters or session
        # This is a simplified version - in production, you'd store this data properly
        
        prediction_data = request.args.get('prediction_data')
        image_path = request.args.get('image_path')
        
        if not prediction_data or not image_path:
            return jsonify({'error': 'Missing required data for report generation'}), 400
        
        # Parse the prediction data
        import urllib.parse
        prediction_data = json.loads(urllib.parse.unquote(prediction_data))
        image_path = urllib.parse.unquote(image_path)
        
        # Generate unique filename
        report_filename = f"plant_disease_report_{uuid.uuid4().hex[:8]}.pdf"
        report_path = os.path.join('reports', report_filename)
        
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        # Generate the PDF report
        if generate_pdf_report(prediction_data, image_path.lstrip('/'), report_path):
            return send_from_directory('reports', report_filename, as_attachment=True)
        else:
            return jsonify({'error': 'Failed to generate report'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Report generation failed: {str(e)}'}), 500

@app.route('/upload/',methods = ['POST','GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        image_path = f'{temp_name}_{image.filename}'
        image.save(image_path)
        print(image_path)
        
        # Get environmental data from form
        soil_ph = request.form.get('soil_ph')
        temperature = request.form.get('temperature')
        humidity = request.form.get('humidity')
        
        # Convert to appropriate types
        environmental_data = {}
        if soil_ph:
            try:
                environmental_data['soil_ph'] = float(soil_ph)
            except ValueError:
                environmental_data['soil_ph'] = None
        if temperature:
            try:
                environmental_data['temperature'] = float(temperature)
            except ValueError:
                environmental_data['temperature'] = None
        if humidity:
            try:
                environmental_data['humidity'] = float(humidity)
            except ValueError:
                environmental_data['humidity'] = None
        
        # Get prediction
        prediction = model_predict(f'./{image_path}')
        
        # Add environmental data to prediction
        prediction['environmental_data'] = environmental_data
        
        # Add yield prediction
        yield_prediction = yield_predictor.predict_yield_impact(
            prediction['primary_disease']['name'],
            prediction['severity'],
            prediction['health_score'],
            prediction['confidence']
        )
        
        # Add harvest recommendation
        harvest_rec = yield_predictor.get_harvest_recommendation(
            prediction['primary_disease']['name'],
            prediction['severity'],
            prediction['health_score']
        )
        
        # Add to prediction result
        prediction['yield_prediction'] = yield_prediction
        prediction['harvest_recommendation'] = harvest_rec
        
        return render_template('home.html',
                             result=True,
                             imagepath=f'/{image_path}', 
                             prediction=prediction)
    
    else:
        return redirect('/')

# Disease Tracking Routes
@app.route('/disease-tracker')
def disease_tracker():
    """Disease tracking dashboard"""
    cases = db_tracker.get_all_cases()
    return render_template('disease_tracker.html', cases=cases)

@app.route('/create-case', methods=['POST'])
def create_case():
    """Create new disease tracking case"""
    try:
        plant_name = request.form.get('plant_name')
        notes = request.form.get('notes', '')
        
        # Get the prediction data from form (this would come from the analysis)
        prediction_data = json.loads(request.form.get('prediction_data'))
        image_path = request.form.get('image_path')
        
        # Create new case
        case_id = db_tracker.create_disease_case(plant_name, prediction_data, image_path, notes)
        
        # Save yield prediction
        yield_prediction = yield_predictor.predict_yield_impact(
            prediction_data['primary_disease']['name'],
            prediction_data['severity'],
            prediction_data['health_score'],
            prediction_data['confidence']
        )
        yield_predictor.save_yield_prediction(case_id, yield_prediction)
        
        return jsonify({'success': True, 'case_id': case_id})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/case/<case_id>')
def view_case(case_id):
    """View individual disease case"""
    case_data = db_tracker.get_disease_case(case_id)
    if not case_data:
        return redirect('/disease-tracker')
    
    # Get treatment effectiveness
    effectiveness = db_tracker.calculate_treatment_effectiveness(case_id)
    
    # Get treatments
    treatments = db_tracker.get_treatments(case_id)
    
    # Get yield predictions
    yield_predictions = yield_predictor.get_yield_predictions(case_id)
    
    return render_template('case_detail.html', 
                         case_data=case_data,
                         effectiveness=effectiveness,
                         treatments=treatments,
                         yield_predictions=yield_predictions)

@app.route('/add-progression/<case_id>', methods=['POST'])
def add_progression(case_id):
    """Add new progression entry"""
    try:
        image = request.files['img']
        treatment_applied = request.form.get('treatment_applied', '')
        notes = request.form.get('notes', '')
        
        # Save new image
        temp_name = f"uploadimages/progression_{uuid.uuid4().hex}"
        image_path = f'{temp_name}_{image.filename}'
        image.save(image_path)
        
        # Get new prediction
        prediction = model_predict(f'./{image_path}')
        
        # Add progression entry
        db_tracker.add_progression_entry(case_id, prediction, image_path, treatment_applied, notes)
        
        return redirect(f'/case/{case_id}')
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/add-treatment/<case_id>', methods=['POST'])
def add_treatment(case_id):
    """Add treatment record"""
    try:
        treatment_name = request.form.get('treatment_name')
        dosage = request.form.get('dosage', '')
        method = request.form.get('method', '')
        cost = float(request.form.get('cost', 0))
        notes = request.form.get('notes', '')
        
        db_tracker.add_treatment_record(case_id, treatment_name, dosage, method, cost, notes)
        
        return redirect(f'/case/{case_id}')
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/create-case-from-analysis', methods=['POST'])
def create_case_from_analysis():
    """Create a case from current analysis results"""
    try:
        data = request.get_json()
        plant_name = data.get('plant_name')
        prediction_data = data.get('prediction_data')
        image_path = data.get('image_path')
        notes = data.get('notes', '')
        
        # Create the case
        case_id = db_tracker.create_disease_case(plant_name, prediction_data, image_path, notes)
        
        # Generate yield prediction
        yield_impact = yield_predictor.predict_yield_impact(
            prediction_data['primary_disease']['name'],
            prediction_data['severity'],
            prediction_data['health_score'],
            prediction_data['confidence']
        )
        
        # Save yield prediction
        yield_predictor.save_yield_prediction(case_id, yield_impact)
        
        return jsonify({'success': True, 'case_id': case_id})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/progression-analysis/<case_id>')
def progression_analysis(case_id):
    """API endpoint for progression analysis data"""
    try:
        case_data = db_tracker.get_disease_case(case_id)
        if not case_data:
            return jsonify({'error': 'Case not found'})
        
        # Prepare progression data for charts
        progression_data = []
        
        # Add initial case data
        initial_case = case_data['case']
        progression_data.append({
            'date': initial_case['created_date'],
            'health_score': initial_case['initial_health_score'],
            'confidence': initial_case['initial_confidence'],
            'severity': initial_case['initial_severity']
        })
        
        # Add progression entries
        for entry in case_data['progression']:
            progression_data.append({
                'date': entry['analysis_date'],
                'health_score': entry['health_score'],
                'confidence': entry['confidence'],
                'severity': entry['severity']
            })
        
        # Calculate treatment effectiveness
        effectiveness = db_tracker.calculate_treatment_effectiveness(case_id)
        
        return jsonify({
            'progression_data': progression_data,
            'effectiveness': effectiveness
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})
        
    
if __name__ == "__main__":
    app.run(debug=True)