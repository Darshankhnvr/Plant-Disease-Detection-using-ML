import json
from datetime import datetime, timedelta
import sqlite3

class YieldPredictor:
    def __init__(self, db_path='disease_tracking.db'):
        self.db_path = db_path
        
        # Disease impact factors (based on research data)
        self.disease_impact_factors = {
            'Apple___Black_rot': {'yield_loss': 0.25, 'quality_loss': 0.40, 'market_value_reduction': 0.30},
            'Apple___Apple_scab': {'yield_loss': 0.15, 'quality_loss': 0.25, 'market_value_reduction': 0.20},
            'Apple___Cedar_apple_rust': {'yield_loss': 0.10, 'quality_loss': 0.15, 'market_value_reduction': 0.15},
            'Tomato___Early_blight': {'yield_loss': 0.20, 'quality_loss': 0.30, 'market_value_reduction': 0.25},
            'Tomato___Late_blight': {'yield_loss': 0.35, 'quality_loss': 0.50, 'market_value_reduction': 0.40},
            'Tomato___Septoria_leaf_spot': {'yield_loss': 0.15, 'quality_loss': 0.20, 'market_value_reduction': 0.18},
            'Corn___Common_rust': {'yield_loss': 0.12, 'quality_loss': 0.15, 'market_value_reduction': 0.10},
            'Potato___Early_blight': {'yield_loss': 0.18, 'quality_loss': 0.25, 'market_value_reduction': 0.20},
            'Potato___Late_blight': {'yield_loss': 0.40, 'quality_loss': 0.60, 'market_value_reduction': 0.50},
            'Grape___Black_rot': {'yield_loss': 0.30, 'quality_loss': 0.45, 'market_value_reduction': 0.35},
            'Grape___Esca_(Black_Measles)': {'yield_loss': 0.25, 'quality_loss': 0.40, 'market_value_reduction': 0.30},
        }
        
        # Crop market prices (USD per kg)
        self.crop_prices = {
            'Apple': 2.50,
            'Tomato': 3.00,
            'Corn': 0.80,
            'Potato': 1.20,
            'Grape': 4.00,
            'Pepper': 3.50,
            'Peach': 3.20,
            'Orange': 2.80,
            'Strawberry': 8.00,
            'Blueberry': 12.00,
            'Cherry': 6.00,
            'Soybean': 1.50,
            'Squash': 2.00,
            'Raspberry': 10.00
        }
        
        # Average yield per plant (kg)
        self.average_yields = {
            'Apple': 45.0,
            'Tomato': 8.0,
            'Corn': 2.5,
            'Potato': 3.0,
            'Grape': 12.0,
            'Pepper': 2.0,
            'Peach': 35.0,
            'Orange': 40.0,
            'Strawberry': 0.5,
            'Blueberry': 2.0,
            'Cherry': 25.0,
            'Soybean': 0.8,
            'Squash': 4.0,
            'Raspberry': 1.5
        }
    
    def extract_crop_type(self, disease_name):
        """Extract crop type from disease name"""
        crop_mapping = {
            'Apple___': 'Apple',
            'Tomato___': 'Tomato',
            'Corn___': 'Corn',
            'Potato___': 'Potato',
            'Grape___': 'Grape',
            'Pepper,_bell___': 'Pepper',
            'Peach___': 'Peach',
            'Orange___': 'Orange',
            'Strawberry___': 'Strawberry',
            'Blueberry___': 'Blueberry',
            'Cherry___': 'Cherry',
            'Soybean___': 'Soybean',
            'Squash___': 'Squash',
            'Raspberry___': 'Raspberry'
        }
        
        for key, crop in crop_mapping.items():
            if disease_name.startswith(key):
                return crop
        return 'Unknown'
    
    def calculate_severity_multiplier(self, severity, health_score):
        """Calculate severity multiplier based on severity and health score"""
        severity_multipliers = {
            'Mild': 0.3,
            'Moderate': 0.7,
            'Severe': 1.0
        }
        
        base_multiplier = severity_multipliers.get(severity, 0.7)
        
        # Adjust based on health score
        health_factor = (100 - health_score) / 100
        
        return base_multiplier * (0.5 + health_factor * 0.5)
    
    def predict_yield_impact(self, disease_name, severity, health_score, confidence, num_plants=1):
        """Predict yield impact based on disease analysis"""
        
        crop_type = self.extract_crop_type(disease_name)
        
        # Get disease impact factors
        disease_factors = self.disease_impact_factors.get(
            disease_name, 
            {'yield_loss': 0.20, 'quality_loss': 0.30, 'market_value_reduction': 0.25}
        )
        
        # Calculate severity multiplier
        severity_multiplier = self.calculate_severity_multiplier(severity, health_score)
        
        # Confidence factor (lower confidence = lower impact prediction reliability)
        confidence_factor = confidence / 100
        
        # Calculate losses
        base_yield_loss = disease_factors['yield_loss'] * severity_multiplier * confidence_factor
        quality_loss = disease_factors['quality_loss'] * severity_multiplier * confidence_factor
        market_value_reduction = disease_factors['market_value_reduction'] * severity_multiplier * confidence_factor
        
        # Get crop data
        crop_price = self.crop_prices.get(crop_type, 2.0)
        average_yield = self.average_yields.get(crop_type, 5.0)
        
        # Calculate economic impact
        expected_yield = average_yield * num_plants
        yield_loss_kg = expected_yield * base_yield_loss
        remaining_yield = expected_yield - yield_loss_kg
        
        # Quality impact on remaining yield
        quality_affected_yield = remaining_yield * quality_loss
        premium_yield = remaining_yield - quality_affected_yield
        
        # Economic calculations
        normal_revenue = expected_yield * crop_price
        actual_revenue = (premium_yield * crop_price) + (quality_affected_yield * crop_price * (1 - market_value_reduction))
        economic_loss = normal_revenue - actual_revenue
        
        return {
            'crop_type': crop_type,
            'expected_yield_kg': round(expected_yield, 2),
            'predicted_yield_loss_kg': round(yield_loss_kg, 2),
            'predicted_yield_loss_percentage': round(base_yield_loss * 100, 1),
            'quality_affected_yield_kg': round(quality_affected_yield, 2),
            'premium_yield_kg': round(premium_yield, 2),
            'normal_revenue_usd': round(normal_revenue, 2),
            'predicted_revenue_usd': round(actual_revenue, 2),
            'economic_loss_usd': round(economic_loss, 2),
            'economic_loss_percentage': round((economic_loss / normal_revenue) * 100, 1) if normal_revenue > 0 else 0,
            'severity_factor': round(severity_multiplier, 2),
            'confidence_factor': round(confidence_factor, 2)
        }
    
    def get_harvest_recommendation(self, disease_name, severity, health_score, days_to_harvest=30):
        """Get harvest timing recommendation"""
        
        if 'healthy' in disease_name.lower():
            return {
                'recommendation': 'Normal harvest timing',
                'urgency': 'Low',
                'action': 'Continue normal care and monitoring'
            }
        
        severity_recommendations = {
            'Mild': {
                'recommendation': 'Monitor closely, harvest as planned',
                'urgency': 'Low',
                'action': 'Apply preventive treatments, continue monitoring'
            },
            'Moderate': {
                'recommendation': 'Consider early harvest if disease progresses',
                'urgency': 'Medium',
                'action': 'Apply treatment immediately, monitor daily'
            },
            'Severe': {
                'recommendation': 'Harvest immediately if possible to minimize losses',
                'urgency': 'High',
                'action': 'Emergency harvest, isolate affected plants'
            }
        }
        
        base_rec = severity_recommendations.get(severity, severity_recommendations['Moderate'])
        
        # Adjust based on health score
        if health_score < 30:
            base_rec['urgency'] = 'Critical'
            base_rec['recommendation'] = 'Emergency harvest required - severe plant stress detected'
        elif health_score < 50 and severity == 'Severe':
            base_rec['urgency'] = 'High'
            
        return base_rec
    
    def save_yield_prediction(self, case_id, prediction_data):
        """Save yield prediction to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        harvest_rec = self.get_harvest_recommendation(
            prediction_data['crop_type'], 
            'Moderate',  # This should come from the disease analysis
            70  # This should come from the disease analysis
        )
        
        cursor.execute('''
            INSERT INTO yield_predictions 
            (case_id, predicted_yield_loss, economic_impact, prediction_date, harvest_recommendation)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            case_id,
            prediction_data['predicted_yield_loss_percentage'],
            prediction_data['economic_loss_usd'],
            datetime.now().isoformat(),
            json.dumps(harvest_rec)
        ))
        
        conn.commit()
        conn.close()
    
    def get_yield_predictions(self, case_id):
        """Get yield predictions for a case"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM yield_predictions 
            WHERE case_id = ? 
            ORDER BY prediction_date DESC
        ''', (case_id,))
        
        predictions = cursor.fetchall()
        conn.close()
        return predictions