import json
from datetime import datetime, timedelta
import sqlite3

class SmartTreatmentAdvisor:
    """
    AI-powered treatment recommendations based on disease, severity, environment, and crop stage
    """
    
    def __init__(self):
        self.treatment_database = self.load_treatment_database()
        self.resistance_patterns = self.load_resistance_patterns()
    
    def load_treatment_database(self):
        """Comprehensive treatment database"""
        return {
            'Tomato___Late_blight': {
                'organic': [
                    {
                        'name': 'Copper Fungicide',
                        'active_ingredient': 'Copper sulfate',
                        'application_rate': '2-3 ml per liter',
                        'frequency': 'Every 7-10 days',
                        'timing': 'Early morning or evening',
                        'effectiveness': 75,
                        'cost_per_hectare': 25,
                        'safety_period': 7,
                        'environmental_impact': 'low'
                    },
                    {
                        'name': 'Baking Soda Spray',
                        'active_ingredient': 'Sodium bicarbonate',
                        'application_rate': '5g per liter + 2ml dish soap',
                        'frequency': 'Every 5-7 days',
                        'timing': 'Early morning',
                        'effectiveness': 60,
                        'cost_per_hectare': 5,
                        'safety_period': 0,
                        'environmental_impact': 'very_low'
                    }
                ],
                'chemical': [
                    {
                        'name': 'Metalaxyl + Mancozeb',
                        'active_ingredient': 'Metalaxyl 8% + Mancozeb 64%',
                        'application_rate': '2.5g per liter',
                        'frequency': 'Every 10-14 days',
                        'timing': 'Before rain or high humidity',
                        'effectiveness': 90,
                        'cost_per_hectare': 45,
                        'safety_period': 14,
                        'environmental_impact': 'medium'
                    }
                ],
                'biological': [
                    {
                        'name': 'Bacillus subtilis',
                        'active_ingredient': 'Bacillus subtilis strain QST 713',
                        'application_rate': '2ml per liter',
                        'frequency': 'Every 7 days',
                        'timing': 'Any time, avoid direct sunlight',
                        'effectiveness': 70,
                        'cost_per_hectare': 35,
                        'safety_period': 0,
                        'environmental_impact': 'very_low'
                    }
                ]
            },
            'Apple___Apple_scab': {
                'organic': [
                    {
                        'name': 'Lime Sulfur',
                        'active_ingredient': 'Calcium polysulfide',
                        'application_rate': '15ml per liter',
                        'frequency': 'Every 10-14 days',
                        'timing': 'Dormant season and early spring',
                        'effectiveness': 80,
                        'cost_per_hectare': 30,
                        'safety_period': 7,
                        'environmental_impact': 'low'
                    }
                ],
                'chemical': [
                    {
                        'name': 'Myclobutanil',
                        'active_ingredient': 'Myclobutanil 10% WP',
                        'application_rate': '1g per liter',
                        'frequency': 'Every 14-21 days',
                        'timing': 'Pre-bloom to fruit development',
                        'effectiveness': 95,
                        'cost_per_hectare': 60,
                        'safety_period': 21,
                        'environmental_impact': 'medium'
                    }
                ]
            }
            # Add more diseases...
        }
    
    def load_resistance_patterns(self):
        """Track resistance patterns for different regions"""
        return {
            'metalaxyl_resistance': {
                'regions': ['North_America', 'Europe'],
                'diseases': ['Tomato___Late_blight', 'Potato___Late_blight'],
                'resistance_level': 'high',
                'alternative_treatments': ['copper_fungicides', 'biological_controls']
            }
        }
    
    def get_smart_recommendations(self, disease_name, severity, environmental_data, crop_stage='vegetative', 
                                farmer_preference='balanced', budget_limit=None, organic_only=False):
        """
        Generate smart treatment recommendations based on multiple factors
        """
        
        base_treatments = self.treatment_database.get(disease_name, {})
        if not base_treatments:
            return self.get_generic_recommendations(disease_name)
        
        recommendations = []
        
        # Filter by farmer preference
        treatment_types = ['organic'] if organic_only else ['organic', 'chemical', 'biological']
        
        for treatment_type in treatment_types:
            if treatment_type in base_treatments:
                for treatment in base_treatments[treatment_type]:
                    # Calculate adjusted effectiveness based on conditions
                    adjusted_treatment = self.adjust_treatment_for_conditions(
                        treatment.copy(), severity, environmental_data, crop_stage
                    )
                    
                    # Apply budget filter
                    if budget_limit and adjusted_treatment['cost_per_hectare'] > budget_limit:
                        continue
                    
                    # Add recommendation score
                    adjusted_treatment['recommendation_score'] = self.calculate_recommendation_score(
                        adjusted_treatment, farmer_preference, severity
                    )
                    
                    adjusted_treatment['treatment_type'] = treatment_type
                    recommendations.append(adjusted_treatment)
        
        # Sort by recommendation score
        recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
        
        # Add integrated pest management advice
        ipm_advice = self.generate_ipm_advice(disease_name, environmental_data, crop_stage)
        
        return {
            'primary_recommendations': recommendations[:3],
            'all_options': recommendations,
            'ipm_advice': ipm_advice,
            'resistance_warnings': self.check_resistance_warnings(disease_name),
            'application_schedule': self.generate_application_schedule(recommendations[0] if recommendations else None)
        }
    
    def adjust_treatment_for_conditions(self, treatment, severity, env_data, crop_stage):
        """Adjust treatment effectiveness and application based on conditions"""
        
        # Adjust for severity
        if severity == 'Severe':
            treatment['frequency'] = treatment['frequency'].replace('10-14', '7-10').replace('14-21', '10-14')
            treatment['effectiveness'] = min(100, treatment['effectiveness'] * 1.1)
        elif severity == 'Mild':
            treatment['effectiveness'] = treatment['effectiveness'] * 0.9
        
        # Adjust for environmental conditions
        if env_data:
            humidity = env_data.get('humidity', 70)
            temperature = env_data.get('temperature', 25)
            
            # High humidity increases fungicide effectiveness
            if humidity > 80 and 'fungicide' in treatment['name'].lower():
                treatment['effectiveness'] = min(100, treatment['effectiveness'] * 1.05)
            
            # Temperature adjustments
            if temperature > 30:
                treatment['timing'] = 'Early morning or late evening (avoid heat)'
            elif temperature < 15:
                treatment['effectiveness'] = treatment['effectiveness'] * 0.95
                treatment['notes'] = treatment.get('notes', '') + ' Reduced effectiveness in cool weather.'
        
        # Adjust for crop stage
        if crop_stage == 'flowering' and 'copper' in treatment['name'].lower():
            treatment['caution'] = 'Reduce concentration during flowering to avoid phytotoxicity'
        
        return treatment
    
    def calculate_recommendation_score(self, treatment, farmer_preference, severity):
        """Calculate recommendation score based on farmer preferences"""
        
        score = treatment['effectiveness']  # Base score
        
        if farmer_preference == 'organic':
            if treatment.get('environmental_impact') in ['very_low', 'low']:
                score += 20
            if treatment.get('safety_period', 0) <= 7:
                score += 10
        
        elif farmer_preference == 'cost_effective':
            # Favor lower cost treatments
            if treatment['cost_per_hectare'] < 30:
                score += 15
            elif treatment['cost_per_hectare'] > 50:
                score -= 10
        
        elif farmer_preference == 'high_efficacy':
            # Favor high effectiveness treatments
            if treatment['effectiveness'] > 85:
                score += 15
        
        # Severity adjustments
        if severity == 'Severe' and treatment['effectiveness'] > 80:
            score += 10
        
        return round(score, 1)
    
    def generate_ipm_advice(self, disease_name, env_data, crop_stage):
        """Generate Integrated Pest Management advice"""
        
        ipm_strategies = {
            'cultural_practices': [],
            'biological_control': [],
            'monitoring': [],
            'prevention': []
        }
        
        if 'blight' in disease_name.lower():
            ipm_strategies['cultural_practices'] = [
                'Improve air circulation by proper plant spacing',
                'Avoid overhead watering, use drip irrigation',
                'Remove infected plant debris immediately',
                'Rotate crops with non-host plants'
            ]
            ipm_strategies['monitoring'] = [
                'Check plants daily during high humidity periods',
                'Monitor weather forecasts for disease-favorable conditions',
                'Use disease prediction models for timing treatments'
            ]
        
        if 'scab' in disease_name.lower():
            ipm_strategies['prevention'] = [
                'Plant resistant varieties when available',
                'Apply dormant season treatments',
                'Maintain proper pruning for air circulation'
            ]
        
        return ipm_strategies
    
    def check_resistance_warnings(self, disease_name):
        """Check for known resistance issues"""
        warnings = []
        
        for resistance_type, info in self.resistance_patterns.items():
            if disease_name in info['diseases']:
                warnings.append({
                    'type': resistance_type,
                    'level': info['resistance_level'],
                    'alternatives': info['alternative_treatments'],
                    'message': f"High resistance to {resistance_type.replace('_', ' ')} reported in some regions"
                })
        
        return warnings
    
    def generate_application_schedule(self, primary_treatment):
        """Generate a detailed application schedule"""
        if not primary_treatment:
            return None
        
        schedule = []
        today = datetime.now()
        
        # Parse frequency (e.g., "Every 7-10 days" -> 8.5 days average)
        frequency_text = primary_treatment['frequency']
        if 'Every' in frequency_text:
            days_text = frequency_text.split('Every ')[1].split(' days')[0]
            if '-' in days_text:
                min_days, max_days = map(int, days_text.split('-'))
                interval_days = (min_days + max_days) / 2
            else:
                interval_days = int(days_text)
        else:
            interval_days = 7  # Default
        
        # Generate 4 applications
        for i in range(4):
            application_date = today + timedelta(days=i * interval_days)
            schedule.append({
                'application_number': i + 1,
                'date': application_date.strftime('%Y-%m-%d'),
                'treatment': primary_treatment['name'],
                'rate': primary_treatment['application_rate'],
                'timing': primary_treatment['timing'],
                'weather_check': 'Check for rain forecast 24 hours before application'
            })
        
        return schedule
    
    def get_generic_recommendations(self, disease_name):
        """Provide generic recommendations for unknown diseases"""
        return {
            'primary_recommendations': [
                {
                    'name': 'Broad Spectrum Fungicide',
                    'active_ingredient': 'Copper-based compound',
                    'application_rate': '2ml per liter',
                    'frequency': 'Every 7-10 days',
                    'effectiveness': 70,
                    'cost_per_hectare': 30,
                    'treatment_type': 'organic'
                }
            ],
            'ipm_advice': {
                'cultural_practices': [
                    'Improve plant spacing for better air circulation',
                    'Remove infected plant material',
                    'Avoid overhead watering'
                ]
            },
            'note': 'Generic recommendations - consult local agricultural extension for specific advice'
        }

# Flask integration
def add_treatment_advisor_routes():
    """Add these routes to your Flask app"""
    
    @app.route('/api/treatment-recommendations', methods=['POST'])
    def get_treatment_recommendations():
        data = request.get_json()
        
        advisor = SmartTreatmentAdvisor()
        recommendations = advisor.get_smart_recommendations(
            disease_name=data.get('disease_name'),
            severity=data.get('severity'),
            environmental_data=data.get('environmental_data'),
            crop_stage=data.get('crop_stage', 'vegetative'),
            farmer_preference=data.get('preference', 'balanced'),
            budget_limit=data.get('budget_limit'),
            organic_only=data.get('organic_only', False)
        )
        
        return jsonify(recommendations)