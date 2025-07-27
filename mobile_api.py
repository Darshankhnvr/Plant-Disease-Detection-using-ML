from flask import request, jsonify
import base64
import io
from PIL import Image
import uuid
import os
from datetime import datetime

class MobileAPI:
    """
    Mobile-optimized API endpoints for smartphone app integration
    """
    
    def __init__(self, app):
        self.app = app
    
    def setup_mobile_routes(self):
        """Setup mobile-specific API routes"""
        
        @self.app.route('/api/mobile/analyze', methods=['POST'])
        def mobile_analyze():
            """Mobile-optimized disease analysis endpoint"""
            try:
                data = request.get_json()
                
                # Handle base64 image from mobile
                image_data = data.get('image')
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                # Decode and save image
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Save temporarily
                temp_filename = f"mobile_temp_{uuid.uuid4().hex}.jpg"
                temp_path = f"uploadimages/{temp_filename}"
                image.save(temp_path)
                
                # Get environmental data
                env_data = {
                    'soil_ph': data.get('soil_ph'),
                    'temperature': data.get('temperature'),
                    'humidity': data.get('humidity')
                }
                
                # Get GPS location
                location = {
                    'latitude': data.get('latitude'),
                    'longitude': data.get('longitude')
                }
                
                # Analyze using your enhanced model
                from enhanced_model_predict import model_predict_enhanced
                prediction = model_predict_enhanced(temp_path, env_data)
                
                # Add location-based insights
                if location['latitude'] and location['longitude']:
                    from weather_integration import WeatherIntegration
                    weather = WeatherIntegration()
                    current_weather = weather.get_current_weather(
                        location['latitude'], location['longitude']
                    )
                    prediction['weather_data'] = current_weather
                
                # Clean up temp file
                os.remove(temp_path)
                
                # Mobile-optimized response
                mobile_response = {
                    'success': True,
                    'disease': {
                        'name': prediction['primary_disease']['name'].replace('___', ' - ').replace('_', ' '),
                        'confidence': prediction['confidence'],
                        'severity': prediction['severity'],
                        'health_score': prediction['health_score']
                    },
                    'treatment': {
                        'immediate_action': prediction['primary_disease']['cure'][:200] + '...',
                        'full_treatment': prediction['primary_disease']['cure']
                    },
                    'environmental_analysis': prediction.get('environmental_analysis', {}),
                    'weather_context': prediction.get('weather_data', {}),
                    'recommendations': self.get_mobile_recommendations(prediction),
                    'urgency_level': self.calculate_urgency_level(prediction)
                }
                
                return jsonify(mobile_response)
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'message': 'Analysis failed. Please try again.'
                }), 500
        
        @self.app.route('/api/mobile/quick-scan', methods=['POST'])
        def mobile_quick_scan():
            """Quick scan for basic disease detection"""
            try:
                data = request.get_json()
                
                # Simplified analysis for quick results
                image_data = data.get('image')
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                temp_filename = f"quick_scan_{uuid.uuid4().hex}.jpg"
                temp_path = f"uploadimages/{temp_filename}"
                image.save(temp_path)
                
                # Quick analysis (image only)
                from enhanced_model_predict import model_predict_enhanced
                prediction = model_predict_enhanced(temp_path, {})
                
                os.remove(temp_path)
                
                # Simplified response
                return jsonify({
                    'success': True,
                    'disease_detected': prediction['confidence'] > 70,
                    'disease_name': prediction['primary_disease']['name'].replace('___', ' - ').replace('_', ' '),
                    'confidence': prediction['confidence'],
                    'severity': prediction['severity'],
                    'immediate_action': self.get_immediate_action(prediction),
                    'needs_detailed_analysis': prediction['confidence'] < 80
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/mobile/field-report', methods=['POST'])
        def mobile_field_report():
            """Create field report from mobile"""
            try:
                data = request.get_json()
                
                # Create comprehensive field report
                report_data = {
                    'timestamp': datetime.now().isoformat(),
                    'location': {
                        'latitude': data.get('latitude'),
                        'longitude': data.get('longitude'),
                        'field_name': data.get('field_name', 'Unknown Field')
                    },
                    'crop_info': {
                        'crop_type': data.get('crop_type'),
                        'variety': data.get('variety'),
                        'planting_date': data.get('planting_date'),
                        'growth_stage': data.get('growth_stage')
                    },
                    'observations': {
                        'disease_symptoms': data.get('symptoms', []),
                        'affected_area_percentage': data.get('affected_percentage'),
                        'symptom_severity': data.get('symptom_severity'),
                        'notes': data.get('notes', '')
                    },
                    'environmental_conditions': {
                        'temperature': data.get('temperature'),
                        'humidity': data.get('humidity'),
                        'soil_ph': data.get('soil_ph'),
                        'recent_weather': data.get('recent_weather')
                    },
                    'images': data.get('images', [])  # Array of base64 images
                }
                
                # Save report to database
                report_id = self.save_field_report(report_data)
                
                # Generate recommendations based on report
                recommendations = self.generate_field_recommendations(report_data)
                
                return jsonify({
                    'success': True,
                    'report_id': report_id,
                    'recommendations': recommendations,
                    'follow_up_actions': self.get_follow_up_actions(report_data)
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/mobile/alerts', methods=['GET'])
        def mobile_alerts():
            """Get mobile-optimized alerts"""
            try:
                user_id = request.args.get('user_id', 'default')
                
                from crop_calendar import CropCalendar
                calendar = CropCalendar()
                alerts = calendar.get_active_alerts(user_id)
                
                # Format for mobile
                mobile_alerts = []
                for alert in alerts:
                    mobile_alerts.append({
                        'id': alert[0],
                        'type': alert[2],
                        'message': alert[4],
                        'priority': alert[6],
                        'date': alert[3],
                        'crop': alert[7],
                        'location': alert[8]
                    })
                
                return jsonify({
                    'success': True,
                    'alerts': mobile_alerts,
                    'alert_count': len(mobile_alerts)
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/mobile/weather-forecast', methods=['GET'])
        def mobile_weather_forecast():
            """Get weather forecast for disease prediction"""
            try:
                lat = float(request.args.get('latitude'))
                lon = float(request.args.get('longitude'))
                
                from weather_integration import WeatherIntegration
                weather = WeatherIntegration()
                
                current = weather.get_current_weather(lat, lon)
                forecast = weather.get_weather_forecast(lat, lon, days=5)
                risk_prediction = weather.predict_disease_risk_from_weather(forecast)
                
                return jsonify({
                    'success': True,
                    'current_weather': current,
                    'forecast': forecast[:10],  # Next 10 periods
                    'disease_risk': risk_prediction,
                    'recommendations': self.get_weather_recommendations(risk_prediction)
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
    
    def get_mobile_recommendations(self, prediction):
        """Get mobile-optimized recommendations"""
        recommendations = []
        
        if prediction['severity'] == 'Severe':
            recommendations.append({
                'type': 'urgent',
                'action': 'Apply treatment immediately',
                'icon': 'warning'
            })
        
        if prediction.get('environmental_enhanced'):
            recommendations.append({
                'type': 'environmental',
                'action': 'Monitor environmental conditions closely',
                'icon': 'thermometer'
            })
        
        recommendations.append({
            'type': 'monitoring',
            'action': 'Check plants daily for changes',
            'icon': 'eye'
        })
        
        return recommendations
    
    def calculate_urgency_level(self, prediction):
        """Calculate urgency level for mobile notifications"""
        if prediction['severity'] == 'Severe' and prediction['confidence'] > 85:
            return 'critical'
        elif prediction['severity'] in ['Moderate', 'Severe']:
            return 'high'
        elif prediction['confidence'] > 70:
            return 'medium'
        else:
            return 'low'
    
    def get_immediate_action(self, prediction):
        """Get immediate action for quick scan"""
        if prediction['severity'] == 'Severe':
            return "Immediate treatment required - isolate affected plants"
        elif prediction['severity'] == 'Moderate':
            return "Apply preventive treatment within 24 hours"
        else:
            return "Monitor closely and maintain good plant hygiene"
    
    def save_field_report(self, report_data):
        """Save field report to database"""
        # Implementation would save to your database
        # Return report ID
        return f"FR_{uuid.uuid4().hex[:8]}"
    
    def generate_field_recommendations(self, report_data):
        """Generate recommendations based on field report"""
        recommendations = []
        
        affected_percentage = report_data['observations'].get('affected_area_percentage', 0)
        
        if affected_percentage > 50:
            recommendations.append("Consider emergency treatment for entire field")
        elif affected_percentage > 20:
            recommendations.append("Treat affected areas and create buffer zones")
        else:
            recommendations.append("Spot treatment and increased monitoring")
        
        return recommendations
    
    def get_follow_up_actions(self, report_data):
        """Get follow-up actions for field report"""
        return [
            "Schedule follow-up inspection in 3-5 days",
            "Monitor weather conditions for treatment timing",
            "Document treatment effectiveness"
        ]
    
    def get_weather_recommendations(self, risk_predictions):
        """Get weather-based recommendations"""
        recommendations = []
        
        for day_risk in risk_predictions[:3]:  # Next 3 days
            if day_risk['overall_risk'] > 70:
                recommendations.append(f"High disease risk on {day_risk['date']} - consider preventive treatment")
            elif day_risk['overall_risk'] > 40:
                recommendations.append(f"Moderate risk on {day_risk['date']} - increase monitoring")
        
        return recommendations

# Usage in your Flask app
def setup_mobile_api(app):
    """Setup mobile API in your Flask app"""
    mobile_api = MobileAPI(app)
    mobile_api.setup_mobile_routes()
    return mobile_api