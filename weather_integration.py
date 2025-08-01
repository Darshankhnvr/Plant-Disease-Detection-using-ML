import requests
import json
from datetime import datetime, timedelta
import sqlite3

class WeatherIntegration:
    """
    Integrate real-time weather data for automatic environmental analysis
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key or "demo_api_key"  # Use demo key for now
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
    def get_current_weather(self, lat, lon):
        """Get current weather conditions"""
        url = f"{self.base_url}/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            if self.api_key == "demo_api_key":
                # Return demo data when no API key is provided
                return {
                    'temperature': 25.0,
                    'humidity': 70,
                    'pressure': 1013,
                    'weather_condition': 'Clear',
                    'description': 'clear sky',
                    'wind_speed': 3.5,
                    'location': 'Demo Location'
                }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if response.status_code == 200:
                return {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'weather_condition': data['weather'][0]['main'],
                    'description': data['weather'][0]['description'],
                    'wind_speed': data['wind']['speed'],
                    'location': data['name']
                }
            else:
                print(f"Weather API error: {data.get('message', 'Unknown error')}")
                return None
        except Exception as e:
            print(f"Weather API error: {e}")
            return None
    
    def get_weather_forecast(self, lat, lon, days=7):
        """Get weather forecast for disease prediction"""
        url = f"{self.base_url}/forecast"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            if self.api_key == "demo_api_key":
                # Return demo forecast data
                from datetime import datetime, timedelta
                forecast = []
                base_date = datetime.now()
                
                for i in range(days * 8):
                    forecast_date = base_date + timedelta(hours=i * 3)
                    forecast.append({
                        'datetime': forecast_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'temperature': 25.0 + (i % 10) - 5,  # Varying temperature
                        'humidity': 70 + (i % 20) - 10,      # Varying humidity
                        'weather': 'Clear' if i % 3 == 0 else 'Clouds',
                        'rain_probability': (i % 30) * 2     # Varying rain probability
                    })
                
                return forecast[:days*8]
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if response.status_code == 200:
                forecast = []
                for item in data['list'][:days*8]:  # 8 forecasts per day (3-hour intervals)
                    forecast.append({
                        'datetime': item['dt_txt'],
                        'temperature': item['main']['temp'],
                        'humidity': item['main']['humidity'],
                        'weather': item['weather'][0]['main'],
                        'rain_probability': item.get('pop', 0) * 100
                    })
                
                return forecast
            else:
                print(f"Forecast API error: {data.get('message', 'Unknown error')}")
                return None
        except Exception as e:
            print(f"Forecast API error: {e}")
            return None
    
    def predict_disease_risk_from_weather(self, forecast_data):
        """Predict disease risk based on weather forecast"""
        risk_predictions = []
        
        for day_data in forecast_data:
            temp = day_data['temperature']
            humidity = day_data['humidity']
            rain_prob = day_data.get('rain_probability', 0)
            
            # Disease risk calculations
            late_blight_risk = 0
            if 15 <= temp <= 25 and humidity >= 80:
                late_blight_risk = min(100, humidity + rain_prob - 80)
            
            powdery_mildew_risk = 0
            if 20 <= temp <= 30 and 40 <= humidity <= 70:
                powdery_mildew_risk = min(100, 100 - abs(humidity - 55))
            
            rust_risk = 0
            if 20 <= temp <= 28 and humidity >= 70:
                rust_risk = min(100, (humidity - 70) * 2 + rain_prob)
            
            risk_predictions.append({
                'date': day_data['datetime'][:10],
                'late_blight_risk': round(late_blight_risk, 1),
                'powdery_mildew_risk': round(powdery_mildew_risk, 1),
                'rust_risk': round(rust_risk, 1),
                'overall_risk': round((late_blight_risk + powdery_mildew_risk + rust_risk) / 3, 1)
            })
        
        return risk_predictions

# Usage example
def add_weather_route_to_flask():
    """Add this to your Flask app"""
    
    @app.route('/api/weather-analysis', methods=['POST'])
    def weather_analysis():
        data = request.get_json()
        lat = data.get('latitude')
        lon = data.get('longitude')
        
        weather = WeatherIntegration()
        current = weather.get_current_weather(lat, lon)
        forecast = weather.get_weather_forecast(lat, lon)
        risk_prediction = weather.predict_disease_risk_from_weather(forecast)
        
        return jsonify({
            'current_weather': current,
            'disease_risk_forecast': risk_prediction
        })