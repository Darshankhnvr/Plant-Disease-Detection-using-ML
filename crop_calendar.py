from datetime import datetime, timedelta
import json
import sqlite3
from yield_predictor import YieldPredictor
from weather_integration import WeatherIntegration
from environmental_correlator import EnvironmentalCorrelator

class CropCalendar:
    """
    Smart crop calendar with disease prevention alerts and treatment scheduling
    """
    
    def __init__(self, db_path='disease_tracking.db'):
        self.db_path = db_path
        self.init_calendar_db()
        self.yield_predictor = YieldPredictor(db_path=db_path)
        self.weather_integration = WeatherIntegration()
        self.environmental_correlator = EnvironmentalCorrelator()
        
        # Crop growth stages and disease susceptibility
        self.crop_stages = {
            'Tomato': {
                'seedling': {'days': 14, 'diseases': ['damping_off'], 'care': 'Keep soil moist, good ventilation'},
                'vegetative': {'days': 30, 'diseases': ['early_blight', 'late_blight'], 'care': 'Regular watering, pruning'},
                'flowering': {'days': 21, 'diseases': ['blossom_end_rot', 'bacterial_spot'], 'care': 'Consistent watering, calcium'},
                'fruiting': {'days': 60, 'diseases': ['late_blight', 'fruit_rot'], 'care': 'Reduce watering, harvest regularly'}
            },
            'Potato': {
                'planting': {'days': 14, 'diseases': ['seed_rot'], 'care': 'Well-drained soil, proper spacing'},
                'emergence': {'days': 21, 'diseases': ['early_blight'], 'care': 'Hill soil, monitor for pests'},
                'tuber_formation': {'days': 35, 'diseases': ['late_blight', 'scab'], 'care': 'Consistent moisture, avoid overwatering'},
                'maturation': {'days': 30, 'diseases': ['storage_rot'], 'care': 'Reduce watering, prepare for harvest'}
            },
            'Apple': {
                'dormant': {'days': 90, 'diseases': ['canker'], 'care': 'Pruning, dormant oil spray'},
                'bud_break': {'days': 21, 'diseases': ['apple_scab'], 'care': 'Fungicide application, sanitation'},
                'bloom': {'days': 14, 'diseases': ['fire_blight'], 'care': 'Avoid overhead watering, bee protection'},
                'fruit_development': {'days': 120, 'diseases': ['cedar_rust', 'black_rot'], 'care': 'Regular monitoring, thinning'}
            },
            'Grape': {
                'dormant': {'days': 60, 'diseases': ['canker'], 'care': 'Pruning, dormant oil spray'},
                'bud_break': {'days': 30, 'diseases': ['powdery_mildew'], 'care': 'Fungicide application, sanitation'},
                'flowering': {'days': 20, 'diseases': ['downy_mildew'], 'care': 'Avoid overhead watering'},
                'fruit_set': {'days': 45, 'diseases': ['black_rot'], 'care': 'Thinning, pest control'},
                'veraison': {'days': 30, 'diseases': ['botrytis_bunch_rot'], 'care': 'Canopy management'},
                'harvest': {'days': 15, 'diseases': [], 'care': 'Monitor sugar levels'}
            },
            'Corn': {
                'planting': {'days': 7, 'diseases': ['seed_rot'], 'care': 'Ensure good soil contact'},
                'emergence': {'days': 14, 'diseases': ['damping_off'], 'care': 'Monitor for cutworms'},
                'vegetative': {'days': 40, 'diseases': ['common_rust', 'gray_leaf_spot'], 'care': 'Fertilize, weed control'},
                'tasseling_silking': {'days': 20, 'diseases': ['smut'], 'care': 'Ensure adequate water for pollination'},
                'grain_fill': {'days': 45, 'diseases': ['ear_rot'], 'care': 'Monitor for pests, maintain moisture'}
            },
            'Pepper': {
                'seedling': {'days': 21, 'diseases': ['damping_off'], 'care': 'Warm, moist conditions'},
                'vegetative': {'days': 40, 'diseases': ['bacterial_spot', 'phytophthora_blight'], 'care': 'Regular watering, support'},
                'flowering': {'days': 30, 'diseases': ['blossom_end_rot'], 'care': 'Consistent watering, calcium'},
                'fruiting': {'days': 60, 'diseases': ['anthracnose', 'cercospora_leaf_spot'], 'care': 'Harvest regularly, pest control'}
            },
            'Strawberry': {
                'planting': {'days': 14, 'diseases': ['root_rot'], 'care': 'Well-drained soil'},
                'vegetative': {'days': 45, 'diseases': ['leaf_spot', 'powdery_mildew'], 'care': 'Runner management, fertilize'},
                'flowering': {'days': 30, 'diseases': ['botrytis_fruit_rot'], 'care': 'Protect from rain, good air circulation'},
                'fruiting': {'days': 30, 'diseases': ['anthracnose_fruit_rot'], 'care': 'Harvest frequently, pest control'}
            },
            'Cherry': {
                'dormant': {'days': 90, 'diseases': ['canker'], 'care': 'Pruning, dormant oil spray'},
                'bud_break': {'days': 25, 'diseases': ['brown_rot'], 'care': 'Fungicide application'},
                'flowering': {'days': 15, 'diseases': ['blossom_blight'], 'care': 'Avoid overhead watering'},
                'fruit_development': {'days': 60, 'diseases': ['cherry_leaf_spot', 'powdery_mildew'], 'care': 'Thinning, pest control'},
                'harvest': {'days': 10, 'diseases': [], 'care': 'Monitor ripeness'}
            }
        }
    
    def init_calendar_db(self):
        """Initialize calendar database tables"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        # Enable WAL mode for better concurrency
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA busy_timeout=30000')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crop_calendar (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                crop_type TEXT NOT NULL,
                planting_date TEXT NOT NULL,
                field_location TEXT,
                variety TEXT,
                expected_harvest TEXT,
                current_stage TEXT,
                created_date TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS calendar_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                calendar_id INTEGER,
                alert_type TEXT NOT NULL,
                alert_date TEXT NOT NULL,
                message TEXT NOT NULL,
                is_read BOOLEAN DEFAULT FALSE,
                priority TEXT DEFAULT 'medium',
                FOREIGN KEY (calendar_id) REFERENCES crop_calendar (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_crop_to_calendar(self, crop_type, planting_date, field_location="", variety="", user_id="default"):
        """Add a new crop to the calendar"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)  # Add timeout
            cursor = conn.cursor()
            
            # Calculate expected harvest date
            expected_harvest = self.yield_predictor.predict_harvest_date(
                crop_type, planting_date, self.crop_stages
            )
            
            cursor.execute('''
                INSERT INTO crop_calendar 
                (user_id, crop_type, planting_date, field_location, variety, expected_harvest, current_stage, created_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, crop_type, planting_date, field_location, variety, 
                expected_harvest, 'seedling', datetime.now().isoformat()
            ))
            
            calendar_id = cursor.lastrowid
            
            # Calculate expected harvest date
            planting_dt = datetime.strptime(planting_date, '%Y-%m-%d') # Define planting_dt here
            expected_harvest = self.yield_predictor.predict_harvest_date(
                crop_type, planting_date, self.crop_stages
            )

            cursor.execute('''
                INSERT INTO crop_calendar
                (user_id, crop_type, planting_date, field_location, variety, expected_harvest, current_stage, created_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, crop_type, planting_date, field_location, variety,
                expected_harvest, 'seedling', datetime.now().isoformat()
            ))

            calendar_id = cursor.lastrowid

            # Generate alerts for this crop using the same connection
            self.generate_crop_alerts_with_connection(cursor, calendar_id, crop_type, planting_dt, field_location)
            
            conn.commit()
            return calendar_id
            
        except sqlite3.OperationalError as e:
            if conn:
                conn.rollback()
            raise Exception(f"Database error: {str(e)}")
        except Exception as e:
            if conn:
                conn.rollback()
            raise Exception(f"Error adding crop: {str(e)}")
        finally:
            if conn:
                conn.close()
    
    def generate_crop_alerts_with_connection(self, cursor, calendar_id, crop_type, planting_date, field_location=""):
        """Generate preventive alerts for crop stages using existing connection"""
        current_date = planting_date
        stages = self.crop_stages.get(crop_type, {})

        # Placeholder for lat/lon - in a real app, resolve from field_location
        # For now, using a generic central US location for demo purposes
        lat = 39.8283
        lon = -98.5795

        # Get weather forecast for the next 7 days
        forecast_data = self.weather_integration.get_weather_forecast(lat, lon, days=7)

        if forecast_data:
            # Check for frost
            frost_dates = self.weather_integration.check_for_frost(forecast_data)
            for f_date in frost_dates:
                cursor.execute('''
                    INSERT INTO calendar_alerts 
                    (calendar_id, alert_type, alert_date, message, priority)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    calendar_id,
                    'frost_warning',
                    f_date,
                    f"Frost warning for {crop_type} at {field_location} on {f_date}. Take protective measures!",
                    'high'
                ))

            # Check for heavy rain
            heavy_rain_dates = self.weather_integration.check_for_heavy_rain(forecast_data)
            for r_date in heavy_rain_dates:
                cursor.execute('''
                    INSERT INTO calendar_alerts 
                    (calendar_id, alert_type, alert_date, message, priority)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    calendar_id,
                    'heavy_rain_warning',
                    r_date,
                    f"Heavy rain expected for {crop_type} at {field_location} on {r_date}. Ensure proper drainage.",
                    'medium'
                ))

            # Check for optimal spraying conditions
            optimal_spraying_times = self.weather_integration.check_optimal_spraying_conditions(forecast_data)
            for s_time in optimal_spraying_times:
                cursor.execute('''
                    INSERT INTO calendar_alerts 
                    (calendar_id, alert_type, alert_date, message, priority)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    calendar_id,
                    'spraying_window',
                    s_time.split(' ')[0], # Use date part only for alert_date
                    f"Optimal spraying window for {crop_type} at {field_location} around {s_time}. Low wind and no significant rain.",
                    'low'
                ))

        # Placeholder for current environmental conditions (e.g., from sensors or manual input)
        # In a real application, this would come from actual data for the field_location
        current_env_conditions = {
            'temperature': 25, # Example value
            'humidity': 70,    # Example value
            'soil_ph': 6.5     # Example value
        }

        # Get environmental recommendations for nutrient/soil management
        env_recommendations = self.environmental_correlator.get_environmental_recommendations(
            crop_type, current_env_conditions
        )

        if env_recommendations and env_recommendations['recommendations']:
            for rec_message in env_recommendations['recommendations']:
                # Filter out generic "within optimal range" messages for alerts
                if "within optimal range" not in rec_message.lower():
                    cursor.execute('''
                        INSERT INTO calendar_alerts 
                        (calendar_id, alert_type, alert_date, message, priority)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        calendar_id,
                        'environmental_recommendation',
                        datetime.now().strftime('%Y-%m-%d'), # Alert for today
                        f"Environmental Recommendation for {crop_type} at {field_location}: {rec_message}",
                        'medium'
                    ))

        for stage_name, stage_info in stages.items():
            # Alert for stage transition
            alert_date = current_date + timedelta(days=stage_info['days'] - 3)  # 3 days before
            
            cursor.execute('''
                INSERT INTO calendar_alerts 
                (calendar_id, alert_type, alert_date, message, priority)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                calendar_id,
                'stage_transition',
                alert_date.strftime('%Y-%m-%d'),
                f"Your {crop_type} is entering {stage_name} stage. {stage_info['care']}",
                'medium'
            ))
            
            # Disease prevention alerts
            for disease in stage_info['diseases']:
                disease_alert_date = current_date + timedelta(days=stage_info['days'] // 2)
                cursor.execute('''
                    INSERT INTO calendar_alerts 
                    (calendar_id, alert_type, alert_date, message, priority)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    calendar_id,
                    'disease_prevention',
                    disease_alert_date.strftime('%Y-%m-%d'),
                    f"Monitor for {disease.replace('_', ' ')} during {stage_name} stage",
                    'high' if 'blight' in disease else 'medium'
                ))
            
            current_date += timedelta(days=stage_info['days'])

    def generate_crop_alerts(self, calendar_id, crop_type, planting_date):
        """Generate preventive alerts for crop stages (legacy method)"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            cursor = conn.cursor()
            self.generate_crop_alerts_with_connection(cursor, calendar_id, crop_type, planting_date)
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def get_active_alerts(self, user_id="default"):
        """Get current active alerts"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            cursor = conn.cursor()
            
            today = datetime.now().strftime('%Y-%m-%d')
            
            cursor.execute('''
                SELECT ca.*, cc.crop_type, cc.field_location
                FROM calendar_alerts ca
                JOIN crop_calendar cc ON ca.calendar_id = cc.id
                WHERE cc.user_id = ? AND ca.alert_date <= ? AND ca.is_read = FALSE
                ORDER BY ca.priority DESC, ca.alert_date ASC
            ''', (user_id, today))
            
            alerts = cursor.fetchall()
            return alerts
        except Exception as e:
            print(f"Error getting alerts: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_crop_schedule(self, user_id="default"):
        """Get upcoming crop activities"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM crop_calendar 
                WHERE user_id = ? 
                ORDER BY planting_date DESC
            ''', (user_id,))
            
            crops = cursor.fetchall()
            
            schedule = []
            for crop in crops:
                crop_id, user_id, crop_type, planting_date, location, variety, expected_harvest, stage, created = crop
                
                # Calculate current stage
                planting_dt = datetime.strptime(planting_date, '%Y-%m-%d')
                days_since_planting = (datetime.now() - planting_dt).days
                
                current_stage = self.calculate_current_stage(crop_type, days_since_planting)
                
                schedule.append({
                    'id': crop_id,
                    'crop_type': crop_type,
                    'location': location,
                    'variety': variety,
                    'planting_date': planting_date,
                    'harvest_date': expected_harvest,
                    'current_stage': current_stage,
                    'days_since_planting': days_since_planting
                })
            
            return schedule
        except Exception as e:
            print(f"Error getting crop schedule: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def calculate_current_stage(self, crop_type, days_since_planting):
        """Calculate current growth stage based on days since planting"""
        stages = self.crop_stages.get(crop_type, {})
        cumulative_days = 0
        
        for stage_name, stage_info in stages.items():
            cumulative_days += stage_info['days']
            if days_since_planting <= cumulative_days:
                return stage_name
        
        return 'mature'
    
    def mark_alert_read(self, alert_id):
        """Mark an alert as read"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            cursor = conn.cursor()
            
            cursor.execute('UPDATE calendar_alerts SET is_read = TRUE WHERE id = ?', (alert_id,))
            
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise Exception(f"Error marking alert as read: {str(e)}")
        finally:
            if conn:
                conn.close()

    def delete_crop_from_calendar(self, crop_id):
        """Delete a crop and its associated alerts from the calendar"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            cursor = conn.cursor()
            
            # Delete associated alerts first
            cursor.execute('DELETE FROM calendar_alerts WHERE calendar_id = ?', (crop_id,))
            
            # Delete the crop entry
            cursor.execute('DELETE FROM crop_calendar WHERE id = ?', (crop_id,))
            
            conn.commit()
            return True
        except Exception as e:
            if conn:
                conn.rollback()
            raise Exception(f"Error deleting crop: {str(e)}")
        finally:
            if conn:
                conn.close()

# Flask integration
def add_calendar_routes():
    """Add these routes to your Flask app"""
    
    @app.route('/crop-calendar')
    def crop_calendar():
        calendar = CropCalendar()
        alerts = calendar.get_active_alerts()
        schedule = calendar.get_crop_schedule()
        
        return render_template('crop_calendar.html', alerts=alerts, schedule=schedule)
    
    @app.route('/add-crop', methods=['POST'])
    def add_crop():
        calendar = CropCalendar()
        
        crop_type = request.form.get('crop_type')
        planting_date = request.form.get('planting_date')
        location = request.form.get('location', '')
        variety = request.form.get('variety', '')
        
        calendar_id = calendar.add_crop_to_calendar(crop_type, planting_date, location, variety)
        
        return jsonify({'success': True, 'calendar_id': calendar_id})