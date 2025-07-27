"""
Integration code to use the hybrid model in your Flask app
This shows how to modify your existing app.py to use both image and environmental data
"""

import tensorflow as tf
import numpy as np
from hybrid_disease_model import HybridDiseaseModel
import os

class HybridDiseasePredictor:
    """
    Wrapper class to handle hybrid predictions in your Flask app
    """
    
    def __init__(self):
        self.hybrid_model = None
        self.fallback_model = None
        self.use_hybrid = False
        
        # Try to load hybrid model first
        self.load_models()
    
    def load_models(self):
        """Load both hybrid and fallback models"""
        
        # Try to load hybrid model
        try:
            if os.path.exists('models/hybrid/hybrid_disease_model.keras'):
                self.hybrid_model = HybridDiseaseModel()
                self.hybrid_model.load_model(
                    'models/hybrid/hybrid_disease_model.keras',
                    'models/hybrid/env_scaler.pkl'
                )
                self.use_hybrid = True
                print("Hybrid model loaded successfully!")
            else:
                print("Hybrid model not found, will use fallback")
        except Exception as e:
            print(f"Could not load hybrid model: {e}")
        
        # Load fallback model (your original model)
        try:
            self.fallback_model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")
            print("Fallback model loaded successfully!")
        except Exception as e:
            print(f"Could not load fallback model: {e}")
    
    def extract_features(self, image_path):
        """Extract features from image (same as your original function)"""
        image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
        feature = tf.keras.utils.img_to_array(image)
        feature = np.array([feature])
        return feature
    
    def predict_with_environment(self, image_path, environmental_data):
        """
        Make prediction using environmental data if available
        Falls back to image-only prediction if hybrid model not available
        """
        
        # Extract image features
        image_features = self.extract_features(image_path)
        
        # Check if we have environmental data and hybrid model
        has_env_data = any(v is not None and v != 0 for v in environmental_data.values())
        
        if self.use_hybrid and has_env_data and self.hybrid_model:
            return self._predict_hybrid(image_features, environmental_data)
        else:
            return self._predict_fallback(image_features, environmental_data)
    
    def _predict_hybrid(self, image_features, environmental_data):
        """Use hybrid model for prediction"""
        
        try:
            # Prepare environmental data
            env_array = [
                environmental_data.get('soil_ph', 6.5),
                environmental_data.get('temperature', 25.0),
                environmental_data.get('humidity', 70.0)
            ]
            
            # Make hybrid prediction
            prediction = self.hybrid_model.predict_hybrid(image_features[0], env_array)
            
            return {
                'prediction': prediction,
                'method': 'hybrid',
                'confidence_boost': 'Environmental data integrated into prediction',
                'environmental_integration': True
            }
            
        except Exception as e:
            print(f"Hybrid prediction failed: {e}")
            return self._predict_fallback(image_features, environmental_data)
    
    def _predict_fallback(self, image_features, environmental_data):
        """Use original image-only model"""
        
        if self.fallback_model is None:
            raise ValueError("No model available for prediction")
        
        prediction = self.fallback_model.predict(image_features)
        
        return {
            'prediction': prediction,
            'method': 'image_only',
            'confidence_boost': 'Using image-only prediction (hybrid model not available)',
            'environmental_integration': False
        }

# Modified model_predict function for your app.py
def model_predict_hybrid(image_path, environmental_data=None):
    """
    Enhanced model_predict function that uses environmental data
    Replace your existing model_predict function with this
    """
    
    # Initialize hybrid predictor (you'd do this once in your app)
    predictor = HybridDiseasePredictor()
    
    # Make prediction
    result = predictor.predict_with_environment(image_path, environmental_data or {})
    prediction = result['prediction']
    
    # Your existing label processing
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
    
    # Primary prediction
    primary_idx = prediction.argmax()
    primary_disease = plant_disease[primary_idx]
    confidence = float(prediction.max() * 100)
    
    # Your existing severity and health score calculation
    def assess_disease_severity(disease_name, confidence):
        if "healthy" in disease_name.lower():
            return "Healthy", "#4caf50"
        if confidence >= 90:
            return "Severe", "#f44336"
        elif confidence >= 70:
            return "Moderate", "#ff9800"
        else:
            return "Mild", "#ffeb3b"
    
    def calculate_health_score(disease_name, confidence, severity):
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
    
    severity, severity_color = assess_disease_severity(primary_disease['name'], confidence)
    health_score = calculate_health_score(primary_disease['name'], confidence, severity)
    
    # Enhanced result with hybrid information
    enhanced_result = {
        'primary_disease': primary_disease,
        'confidence': round(confidence, 1),
        'severity': severity,
        'severity_color': severity_color,
        'health_score': health_score,
        'prediction_method': result['method'],
        'environmental_integration': result['environmental_integration'],
        'model_enhancement': result['confidence_boost']
    }
    
    return enhanced_result

# Example of how to modify your Flask route
def example_flask_route_modification():
    """
    Example of how to modify your /upload/ route to use the hybrid model
    """
    
    # In your app.py, replace the model_predict call with:
    
    # OLD CODE:
    # prediction = model_predict(f'./{image_path}')
    
    # NEW CODE:
    prediction = model_predict_hybrid(f'./{image_path}', environmental_data)
    
    # The rest of your code remains the same!
    # The prediction will now include environmental integration information

if __name__ == "__main__":
    print("=== Hybrid Model Integration Example ===")
    
    # Test the hybrid predictor
    predictor = HybridDiseasePredictor()
    
    print(f"Hybrid model available: {predictor.use_hybrid}")
    print(f"Fallback model available: {predictor.fallback_model is not None}")
    
    # Example environmental data
    test_env_data = {
        'soil_ph': 6.5,
        'temperature': 25.0,
        'humidity': 80.0
    }
    
    print("\nExample environmental data:", test_env_data)
    print("\nTo integrate this into your app:")
    print("1. Replace your model_predict function with model_predict_hybrid")
    print("2. Train the hybrid model with your actual dataset")
    print("3. The system will automatically use hybrid prediction when environmental data is available")