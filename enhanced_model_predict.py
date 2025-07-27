"""
Enhanced model_predict function that integrates environmental data
Replace your existing model_predict function with this
"""

import tensorflow as tf
import numpy as np
import json
from practical_hybrid_solution import PracticalHybridPredictor
import os

# Global predictor instance (initialize once)
global_predictor = None

def initialize_hybrid_predictor():
    """Initialize the hybrid predictor (call this once when your app starts)"""
    global global_predictor
    
    if global_predictor is None:
        global_predictor = PracticalHybridPredictor()
        
        # Load existing image model
        if global_predictor.load_existing_model():
            print("✓ Image model loaded")
        else:
            print("✗ Could not load image model")
            return False
        
        # Try to load environmental classifier
        if os.path.exists('models/environmental/env_classifier.pkl'):
            if global_predictor.load_environmental_classifier(
                'models/environmental/env_scaler.pkl',
                'models/environmental/env_classifier.pkl'
            ):
                print("✓ Environmental classifier loaded")
            else:
                print("⚠ Creating new environmental classifier")
                global_predictor.create_environmental_classifier()
        else:
            print("⚠ Creating environmental classifier for first time")
            global_predictor.create_environmental_classifier()
            os.makedirs('models/environmental', exist_ok=True)
            global_predictor.save_environmental_classifier(
                'models/environmental/env_scaler.pkl',
                'models/environmental/env_classifier.pkl'
            )
    
    return True

def calculate_confidence_score(prediction):
    """Calculate confidence percentage"""
    return float(prediction.max() * 100)

def assess_disease_severity(disease_name, confidence):
    """Assess disease severity based on confidence and disease type"""
    if "healthy" in disease_name.lower():
        return "Healthy", "#4caf50"
    
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
    
    disease_penalties = {
        "blight": 40, "rot": 35, "rust": 25, "spot": 20,
        "mildew": 15, "virus": 45, "bacterial": 30
    }
    
    penalty = 25
    for disease_type, disease_penalty in disease_penalties.items():
        if disease_type in disease_name.lower():
            penalty = disease_penalty
            break
    
    severity_multipliers = {"Mild": 0.5, "Moderate": 0.8, "Severe": 1.2}
    final_penalty = penalty * severity_multipliers.get(severity, 0.8)
    health_score = max(0, base_score - final_penalty)
    
    return round(health_score)

def model_predict_enhanced(image_path, environmental_data=None):
    """
    Enhanced model prediction that uses both image and environmental data
    
    Args:
        image_path: Path to the image file
        environmental_data: Dict with 'soil_ph', 'temperature', 'humidity' keys
    
    Returns:
        Enhanced prediction result with environmental analysis
    """
    
    # Initialize predictor if not done already
    if not initialize_hybrid_predictor():
        raise ValueError("Could not initialize hybrid predictor")
    
    # Your existing labels
    label = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
             'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
             'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight',
             'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
             'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
             'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
             'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
             'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
             'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
             'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
             'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    
    # Load plant disease data
    with open("plant_disease.json", 'r') as file:
        plant_disease = json.load(file)
    
    # Make hybrid prediction
    hybrid_result = global_predictor.predict_hybrid(image_path, environmental_data or {})
    
    # Extract prediction data
    prediction = hybrid_result['final_prediction']
    primary_idx = hybrid_result['predicted_class']
    confidence = hybrid_result['confidence'] * 100  # Convert to percentage
    
    # Get disease information
    primary_disease = plant_disease[primary_idx]
    
    # Calculate severity and health score
    severity, severity_color = assess_disease_severity(primary_disease['name'], confidence)
    health_score = calculate_health_score(primary_disease['name'], confidence, severity)
    
    # Create enhanced result
    result = {
        'primary_disease': primary_disease,
        'confidence': round(confidence, 1),
        'severity': severity,
        'severity_color': severity_color,
        'health_score': health_score,
        'has_multiple': False,
        
        # Enhanced information
        'prediction_method': hybrid_result['method'],
        'environmental_enhanced': hybrid_result.get('environmental_boost', False),
        'confidence_adjustment': round(hybrid_result.get('confidence_adjustment', 0) * 100, 1),
        'predictions_agree': hybrid_result.get('predictions_agree', True),
    }
    
    # Add environmental analysis details if available
    if hybrid_result.get('environmental_boost'):
        result['environmental_analysis'] = {
            'image_confidence': round(hybrid_result['image_confidence'] * 100, 1),
            'env_confidence': round(hybrid_result['env_confidence'] * 100, 1),
            'combined_confidence': round(confidence, 1),
            'agreement': hybrid_result['predictions_agree'],
            'enhancement_note': _get_enhancement_note(hybrid_result)
        }
    
    return result

def _get_enhancement_note(hybrid_result):
    """Generate a human-readable note about the environmental enhancement"""
    
    if not hybrid_result.get('environmental_boost'):
        return "Image-only prediction (no environmental data provided)"
    
    confidence_change = hybrid_result.get('confidence_adjustment', 0) * 100
    agree = hybrid_result.get('predictions_agree', True)
    
    if agree:
        if confidence_change > 2:
            return "Environmental conditions strongly support this diagnosis"
        elif confidence_change > 0:
            return "Environmental conditions support this diagnosis"
        else:
            return "Environmental conditions are neutral for this diagnosis"
    else:
        return "Environmental conditions suggest a different diagnosis - consider retesting"

# Example of how to modify your Flask route
def example_flask_integration():
    """
    Example of how to integrate this into your Flask app
    """
    
    # In your app.py, in the /upload/ route, replace:
    
    # OLD CODE:
    # prediction = model_predict(f'./{image_path}')
    
    # NEW CODE:
    # prediction = model_predict_enhanced(f'./{image_path}', environmental_data)
    
    # That's it! The rest of your code works the same way
    
    pass

# Test function
def test_enhanced_prediction():
    """Test the enhanced prediction function"""
    
    print("Testing enhanced prediction...")
    
    # Test environmental data
    test_env_data = {
        'soil_ph': 6.2,
        'temperature': 20.0,
        'humidity': 95.0
    }
    
    print(f"Test environmental data: {test_env_data}")
    
    # Note: This would need a real image file to test fully
    # For now, just test the initialization
    success = initialize_hybrid_predictor()
    print(f"Hybrid predictor initialization: {'✓ Success' if success else '✗ Failed'}")
    
    if success:
        print("\n✓ Enhanced prediction system ready!")
        print("Your Flask app can now use environmental data for better accuracy!")
        
        print("\nFeatures added:")
        print("- Combines image CNN with environmental classifier")
        print("- Adjusts confidence based on environmental correlation")
        print("- Shows agreement between image and environmental predictions")
        print("- Provides detailed analysis of enhancement")

if __name__ == "__main__":
    test_enhanced_prediction()