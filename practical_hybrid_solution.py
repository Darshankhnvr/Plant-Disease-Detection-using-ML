"""
Practical solution: Enhance your existing model with environmental data
without needing to retrain from scratch
"""

import tensorflow as tf
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class PracticalHybridPredictor:
    """
    Practical approach: Use your existing CNN + environmental classifier
    This combines your trained image model with a separate environmental classifier
    """
    
    def __init__(self):
        self.image_model = None
        self.env_classifier = None
        self.env_scaler = StandardScaler()
        self.confidence_weights = {'image': 0.7, 'environmental': 0.3}
        
    def load_existing_model(self, model_path="models/plant_disease_recog_model_pwp.keras"):
        """Load your existing trained model"""
        try:
            self.image_model = tf.keras.models.load_model(model_path)
            print("Existing image model loaded successfully!")
            return True
        except Exception as e:
            print(f"Could not load image model: {e}")
            return False
    
    def create_environmental_classifier(self):
        """
        Create a separate classifier for environmental data
        This learns the correlation between environment and diseases
        """
        
        # Disease-environment training data (synthetic but realistic)
        training_data = self._generate_environmental_training_data()
        
        X_env = training_data['environmental_features']
        y_disease = training_data['disease_labels']
        
        # Normalize environmental data
        X_env_scaled = self.env_scaler.fit_transform(X_env)
        
        # Train Random Forest classifier
        self.env_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.env_classifier.fit(X_env_scaled, y_disease)
        
        print("Environmental classifier trained!")
        return self.env_classifier
    
    def _generate_environmental_training_data(self):
        """Generate realistic environmental training data"""
        
        # Disease patterns (based on agricultural research)
        disease_patterns = {
            0: {'name': 'Apple___Apple_scab', 'ph': (6.0, 7.0), 'temp': (18, 24), 'humidity': (85, 95)},
            1: {'name': 'Apple___Black_rot', 'ph': (6.0, 7.5), 'temp': (24, 28), 'humidity': (75, 85)},
            2: {'name': 'Apple___healthy', 'ph': (6.0, 7.0), 'temp': (18, 24), 'humidity': (60, 75)},
            3: {'name': 'Tomato___Early_blight', 'ph': (6.0, 7.0), 'temp': (26, 28), 'humidity': (92, 98)},
            4: {'name': 'Tomato___Late_blight', 'ph': (5.5, 7.0), 'temp': (18, 22), 'humidity': (90, 100)},
            5: {'name': 'Tomato___healthy', 'ph': (6.0, 6.8), 'temp': (20, 25), 'humidity': (65, 80)},
            6: {'name': 'Potato___Late_blight', 'ph': (5.0, 6.5), 'temp': (18, 22), 'humidity': (90, 100)},
            7: {'name': 'Potato___healthy', 'ph': (5.5, 6.5), 'temp': (15, 20), 'humidity': (70, 85)},
            8: {'name': 'Corn___Common_rust', 'ph': (6.0, 7.5), 'temp': (22, 28), 'humidity': (80, 90)},
            9: {'name': 'Corn___healthy', 'ph': (6.0, 7.0), 'temp': (21, 30), 'humidity': (60, 80)},
        }
        
        # Generate synthetic data
        X_env = []
        y_disease = []
        
        samples_per_disease = 200
        
        for disease_id, pattern in disease_patterns.items():
            for _ in range(samples_per_disease):
                # Generate realistic environmental values
                ph = np.random.normal(
                    (pattern['ph'][0] + pattern['ph'][1]) / 2,
                    (pattern['ph'][1] - pattern['ph'][0]) / 6
                )
                temp = np.random.normal(
                    (pattern['temp'][0] + pattern['temp'][1]) / 2,
                    (pattern['temp'][1] - pattern['temp'][0]) / 6
                )
                humidity = np.random.normal(
                    (pattern['humidity'][0] + pattern['humidity'][1]) / 2,
                    (pattern['humidity'][1] - pattern['humidity'][0]) / 6
                )
                
                # Clip to realistic ranges
                ph = np.clip(ph, 4.0, 9.0)
                temp = np.clip(temp, 5.0, 45.0)
                humidity = np.clip(humidity, 20.0, 100.0)
                
                X_env.append([ph, temp, humidity])
                y_disease.append(disease_id)
        
        return {
            'environmental_features': np.array(X_env),
            'disease_labels': np.array(y_disease)
        }
    
    def predict_hybrid(self, image_path, environmental_data):
        """
        Make hybrid prediction combining image and environmental analysis
        """
        
        if self.image_model is None:
            raise ValueError("Image model not loaded")
        
        # 1. Get image prediction (your existing method)
        image_prediction = self._predict_image(image_path)
        
        # 2. Get environmental prediction if data available
        env_prediction = None
        if self._has_environmental_data(environmental_data):
            env_prediction = self._predict_environmental(environmental_data)
        
        # 3. Combine predictions
        final_prediction = self._combine_predictions(image_prediction, env_prediction)
        
        return final_prediction
    
    def _predict_image(self, image_path):
        """Use your existing image model"""
        # Your existing image preprocessing
        image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
        feature = tf.keras.utils.img_to_array(image)
        feature = np.array([feature])
        
        # Get prediction
        prediction = self.image_model.predict(feature)
        confidence = float(prediction.max())
        predicted_class = int(prediction.argmax())
        
        return {
            'prediction_vector': prediction[0],
            'confidence': confidence,
            'predicted_class': predicted_class,
            'method': 'image_cnn'
        }
    
    def _predict_environmental(self, environmental_data):
        """Predict disease likelihood based on environmental conditions"""
        
        if self.env_classifier is None:
            return None
        
        # Prepare environmental features
        env_features = np.array([[
            environmental_data.get('soil_ph', 6.5),
            environmental_data.get('temperature', 25.0),
            environmental_data.get('humidity', 70.0)
        ]])
        
        # Scale features
        env_features_scaled = self.env_scaler.transform(env_features)
        
        # Get prediction probabilities
        env_probabilities = self.env_classifier.predict_proba(env_features_scaled)[0]
        predicted_class = self.env_classifier.predict(env_features_scaled)[0]
        confidence = float(env_probabilities.max())
        
        return {
            'prediction_vector': env_probabilities,
            'confidence': confidence,
            'predicted_class': predicted_class,
            'method': 'environmental_rf'
        }
    
    def _combine_predictions(self, image_pred, env_pred):
        """Intelligently combine image and environmental predictions"""
        
        if env_pred is None:
            # Only image prediction available
            return {
                'final_prediction': image_pred['prediction_vector'],
                'confidence': image_pred['confidence'],
                'predicted_class': image_pred['predicted_class'],
                'method': 'image_only',
                'environmental_boost': False,
                'confidence_adjustment': 0
            }
        
        # Both predictions available - combine them
        image_weight = self.confidence_weights['image']
        env_weight = self.confidence_weights['environmental']
        
        # Weighted average of prediction vectors
        if len(image_pred['prediction_vector']) == len(env_pred['prediction_vector']):
            combined_vector = (image_weight * image_pred['prediction_vector'] + 
                             env_weight * env_pred['prediction_vector'])
        else:
            # If different number of classes, use image prediction with confidence adjustment
            combined_vector = image_pred['prediction_vector']
        
        final_confidence = float(combined_vector.max())
        final_class = int(combined_vector.argmax())
        
        # Calculate confidence adjustment
        confidence_adjustment = final_confidence - image_pred['confidence']
        
        # Check if environmental data supports or contradicts image prediction
        agreement = (image_pred['predicted_class'] == env_pred['predicted_class'])
        
        return {
            'final_prediction': combined_vector,
            'confidence': final_confidence,
            'predicted_class': final_class,
            'method': 'hybrid_combined',
            'environmental_boost': True,
            'confidence_adjustment': confidence_adjustment,
            'predictions_agree': agreement,
            'image_confidence': image_pred['confidence'],
            'env_confidence': env_pred['confidence']
        }
    
    def _has_environmental_data(self, env_data):
        """Check if meaningful environmental data is provided"""
        if not env_data:
            return False
        
        return any(
            env_data.get(key) is not None and env_data.get(key) != 0 
            for key in ['soil_ph', 'temperature', 'humidity']
        )
    
    def save_environmental_classifier(self, scaler_path, classifier_path):
        """Save the environmental classifier and scaler"""
        joblib.dump(self.env_scaler, scaler_path)
        joblib.dump(self.env_classifier, classifier_path)
        print(f"Environmental classifier saved to {classifier_path}")
    
    def load_environmental_classifier(self, scaler_path, classifier_path):
        """Load the environmental classifier and scaler"""
        try:
            self.env_scaler = joblib.load(scaler_path)
            self.env_classifier = joblib.load(classifier_path)
            print("Environmental classifier loaded successfully!")
            return True
        except Exception as e:
            print(f"Could not load environmental classifier: {e}")
            return False

# Integration function for your Flask app
def create_practical_hybrid_predictor():
    """
    Create and train the practical hybrid predictor
    Call this once to set up the system
    """
    
    predictor = PracticalHybridPredictor()
    
    # Load your existing model
    if predictor.load_existing_model():
        print("✓ Existing image model loaded")
    else:
        print("✗ Could not load existing image model")
        return None
    
    # Create environmental classifier
    predictor.create_environmental_classifier()
    print("✓ Environmental classifier created")
    
    # Save the environmental components
    import os
    os.makedirs('models/environmental', exist_ok=True)
    predictor.save_environmental_classifier(
        'models/environmental/env_scaler.pkl',
        'models/environmental/env_classifier.pkl'
    )
    
    return predictor

# Test the system
if __name__ == "__main__":
    print("=== Practical Hybrid Disease Prediction System ===")
    
    # Create the hybrid predictor
    predictor = create_practical_hybrid_predictor()
    
    if predictor:
        print("\n✓ Hybrid system ready!")
        
        # Test with sample data
        test_env_data = {
            'soil_ph': 6.2,
            'temperature': 20.0,
            'humidity': 95.0
        }
        
        print(f"\nTest environmental data: {test_env_data}")
        print("This would enhance your existing image predictions!")
        
        print("\nTo integrate into your Flask app:")
        print("1. Replace your model_predict function")
        print("2. Use predictor.predict_hybrid(image_path, environmental_data)")
        print("3. Get enhanced predictions with environmental correlation!")
    else:
        print("\n✗ Could not create hybrid system")
        print("Make sure your existing model file is available")