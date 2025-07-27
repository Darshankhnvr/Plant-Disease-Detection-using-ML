import json
from typing import Dict, Optional, Tuple

class EnvironmentalCorrelator:
    """
    Correlates environmental conditions with disease likelihood and accuracy
    to improve prediction confidence and provide environmental insights
    """
    
    def __init__(self):
        # Disease-environment correlation data based on agricultural research
        self.disease_environmental_factors = {
            'Apple___Apple_scab': {
                'optimal_conditions': {'temp_range': (15, 25), 'humidity_range': (80, 100), 'ph_range': (6.0, 7.0)},
                'high_risk_conditions': {'temp_range': (18, 24), 'humidity_range': (85, 95), 'ph_range': (6.5, 7.5)},
                'environmental_weight': 0.85  # How much environment affects this disease
            },
            'Apple___Black_rot': {
                'optimal_conditions': {'temp_range': (20, 30), 'humidity_range': (70, 90), 'ph_range': (6.0, 7.5)},
                'high_risk_conditions': {'temp_range': (24, 28), 'humidity_range': (75, 85), 'ph_range': (6.5, 7.0)},
                'environmental_weight': 0.75
            },
            'Tomato___Early_blight': {
                'optimal_conditions': {'temp_range': (24, 29), 'humidity_range': (90, 100), 'ph_range': (6.0, 7.0)},
                'high_risk_conditions': {'temp_range': (26, 28), 'humidity_range': (92, 98), 'ph_range': (6.2, 6.8)},
                'environmental_weight': 0.90
            },
            'Tomato___Late_blight': {
                'optimal_conditions': {'temp_range': (15, 25), 'humidity_range': (85, 100), 'ph_range': (5.5, 7.0)},
                'high_risk_conditions': {'temp_range': (18, 22), 'humidity_range': (90, 100), 'ph_range': (6.0, 6.5)},
                'environmental_weight': 0.95
            },
            'Potato___Late_blight': {
                'optimal_conditions': {'temp_range': (15, 25), 'humidity_range': (85, 100), 'ph_range': (5.0, 6.5)},
                'high_risk_conditions': {'temp_range': (18, 22), 'humidity_range': (90, 100), 'ph_range': (5.5, 6.0)},
                'environmental_weight': 0.95
            },
            'Corn___Common_rust': {
                'optimal_conditions': {'temp_range': (20, 30), 'humidity_range': (70, 95), 'ph_range': (6.0, 7.5)},
                'high_risk_conditions': {'temp_range': (22, 28), 'humidity_range': (80, 90), 'ph_range': (6.5, 7.0)},
                'environmental_weight': 0.80
            },
            'Grape___Black_rot': {
                'optimal_conditions': {'temp_range': (20, 30), 'humidity_range': (75, 95), 'ph_range': (6.0, 7.0)},
                'high_risk_conditions': {'temp_range': (24, 28), 'humidity_range': (80, 90), 'ph_range': (6.2, 6.8)},
                'environmental_weight': 0.85
            },
            'Cherry___Powdery_mildew': {
                'optimal_conditions': {'temp_range': (20, 27), 'humidity_range': (40, 70), 'ph_range': (6.0, 7.5)},
                'high_risk_conditions': {'temp_range': (22, 25), 'humidity_range': (50, 65), 'ph_range': (6.5, 7.0)},
                'environmental_weight': 0.70
            }
        }
        
        # Healthy plant optimal conditions
        self.healthy_conditions = {
            'Apple': {'temp_range': (18, 24), 'humidity_range': (60, 75), 'ph_range': (6.0, 7.0)},
            'Tomato': {'temp_range': (20, 25), 'humidity_range': (65, 80), 'ph_range': (6.0, 6.8)},
            'Potato': {'temp_range': (15, 20), 'humidity_range': (70, 85), 'ph_range': (5.5, 6.5)},
            'Corn': {'temp_range': (21, 30), 'humidity_range': (60, 80), 'ph_range': (6.0, 7.0)},
            'Grape': {'temp_range': (20, 30), 'humidity_range': (50, 70), 'ph_range': (6.0, 7.0)},
            'Cherry': {'temp_range': (18, 24), 'humidity_range': (60, 75), 'ph_range': (6.0, 7.0)}
        }
    
    def calculate_environmental_risk_score(self, disease_name: str, environmental_data: Dict) -> Dict:
        """
        Calculate environmental risk score for a specific disease
        Returns risk assessment and confidence adjustment
        """
        if disease_name not in self.disease_environmental_factors:
            return {
                'environmental_risk': 'unknown',
                'risk_score': 0.5,
                'confidence_adjustment': 0,
                'risk_factors': []
            }
        
        disease_factors = self.disease_environmental_factors[disease_name]
        risk_factors = []
        risk_score = 0.0
        total_factors = 0
        
        # Check temperature
        if environmental_data.get('temperature') is not None:
            temp = environmental_data['temperature']
            high_risk_temp = disease_factors['high_risk_conditions']['temp_range']
            
            if high_risk_temp[0] <= temp <= high_risk_temp[1]:
                risk_score += 0.8
                risk_factors.append(f"Temperature ({temp}°C) is in high-risk range for this disease")
            elif disease_factors['optimal_conditions']['temp_range'][0] <= temp <= disease_factors['optimal_conditions']['temp_range'][1]:
                risk_score += 0.6
                risk_factors.append(f"Temperature ({temp}°C) supports disease development")
            else:
                risk_score += 0.2
                risk_factors.append(f"Temperature ({temp}°C) is less favorable for disease")
            total_factors += 1
        
        # Check humidity
        if environmental_data.get('humidity') is not None:
            humidity = environmental_data['humidity']
            high_risk_humidity = disease_factors['high_risk_conditions']['humidity_range']
            
            if high_risk_humidity[0] <= humidity <= high_risk_humidity[1]:
                risk_score += 0.9
                risk_factors.append(f"Humidity ({humidity}%) is in high-risk range for this disease")
            elif disease_factors['optimal_conditions']['humidity_range'][0] <= humidity <= disease_factors['optimal_conditions']['humidity_range'][1]:
                risk_score += 0.7
                risk_factors.append(f"Humidity ({humidity}%) supports disease development")
            else:
                risk_score += 0.3
                risk_factors.append(f"Humidity ({humidity}%) is less favorable for disease")
            total_factors += 1
        
        # Check pH
        if environmental_data.get('soil_ph') is not None:
            ph = environmental_data['soil_ph']
            high_risk_ph = disease_factors['high_risk_conditions']['ph_range']
            
            if high_risk_ph[0] <= ph <= high_risk_ph[1]:
                risk_score += 0.7
                risk_factors.append(f"Soil pH ({ph}) is in high-risk range for this disease")
            elif disease_factors['optimal_conditions']['ph_range'][0] <= ph <= disease_factors['optimal_conditions']['ph_range'][1]:
                risk_score += 0.5
                risk_factors.append(f"Soil pH ({ph}) supports disease development")
            else:
                risk_score += 0.2
                risk_factors.append(f"Soil pH ({ph}) is less favorable for disease")
            total_factors += 1
        
        if total_factors == 0:
            return {
                'environmental_risk': 'unknown',
                'risk_score': 0.5,
                'confidence_adjustment': 0,
                'risk_factors': ['No environmental data provided']
            }
        
        # Calculate average risk score
        avg_risk_score = risk_score / total_factors
        
        # Determine risk level
        if avg_risk_score >= 0.7:
            risk_level = 'high'
        elif avg_risk_score >= 0.5:
            risk_level = 'moderate'
        else:
            risk_level = 'low'
        
        # Calculate confidence adjustment based on environmental correlation
        environmental_weight = disease_factors['environmental_weight']
        confidence_adjustment = (avg_risk_score - 0.5) * environmental_weight * 20  # Max ±20% adjustment
        
        return {
            'environmental_risk': risk_level,
            'risk_score': round(avg_risk_score, 2),
            'confidence_adjustment': round(confidence_adjustment, 1),
            'risk_factors': risk_factors,
            'environmental_weight': environmental_weight
        }
    
    def enhance_prediction_with_environment(self, prediction_data: Dict, environmental_data: Dict) -> Dict:
        """
        Enhance the original prediction with environmental correlation analysis
        """
        disease_name = prediction_data['primary_disease']['name']
        original_confidence = prediction_data['confidence']
        
        # Calculate environmental risk
        env_analysis = self.calculate_environmental_risk_score(disease_name, environmental_data)
        
        # Adjust confidence based on environmental factors
        adjusted_confidence = original_confidence + env_analysis['confidence_adjustment']
        adjusted_confidence = max(0, min(100, adjusted_confidence))  # Keep within 0-100 range
        
        # Add environmental analysis to prediction
        prediction_data['environmental_analysis'] = env_analysis
        prediction_data['original_confidence'] = original_confidence
        prediction_data['adjusted_confidence'] = round(adjusted_confidence, 1)
        
        # Update main confidence if environmental data strongly supports/contradicts
        if abs(env_analysis['confidence_adjustment']) > 5:
            prediction_data['confidence'] = adjusted_confidence
            prediction_data['confidence_enhanced'] = True
        else:
            prediction_data['confidence_enhanced'] = False
        
        return prediction_data
    
    def get_environmental_recommendations(self, crop_type: str, current_conditions: Dict) -> Dict:
        """
        Provide recommendations for optimal environmental conditions
        """
        if crop_type not in self.healthy_conditions:
            return {'recommendations': ['Environmental data not available for this crop type']}
        
        optimal = self.healthy_conditions[crop_type]
        recommendations = []
        
        # Temperature recommendations
        if current_conditions.get('temperature') is not None:
            temp = current_conditions['temperature']
            opt_temp = optimal['temp_range']
            if temp < opt_temp[0]:
                recommendations.append(f"Temperature is low ({temp}°C). Optimal range: {opt_temp[0]}-{opt_temp[1]}°C")
            elif temp > opt_temp[1]:
                recommendations.append(f"Temperature is high ({temp}°C). Optimal range: {opt_temp[0]}-{opt_temp[1]}°C")
            else:
                recommendations.append(f"Temperature ({temp}°C) is within optimal range")
        
        # Humidity recommendations
        if current_conditions.get('humidity') is not None:
            humidity = current_conditions['humidity']
            opt_humidity = optimal['humidity_range']
            if humidity < opt_humidity[0]:
                recommendations.append(f"Humidity is low ({humidity}%). Consider increasing to {opt_humidity[0]}-{opt_humidity[1]}%")
            elif humidity > opt_humidity[1]:
                recommendations.append(f"Humidity is high ({humidity}%). Consider reducing to {opt_humidity[0]}-{opt_humidity[1]}%")
            else:
                recommendations.append(f"Humidity ({humidity}%) is within optimal range")
        
        # pH recommendations
        if current_conditions.get('soil_ph') is not None:
            ph = current_conditions['soil_ph']
            opt_ph = optimal['ph_range']
            if ph < opt_ph[0]:
                recommendations.append(f"Soil pH is low ({ph}). Consider adding lime to reach {opt_ph[0]}-{opt_ph[1]}")
            elif ph > opt_ph[1]:
                recommendations.append(f"Soil pH is high ({ph}). Consider adding sulfur to reach {opt_ph[0]}-{opt_ph[1]}")
            else:
                recommendations.append(f"Soil pH ({ph}) is within optimal range")
        
        return {
            'recommendations': recommendations,
            'optimal_conditions': optimal
        }