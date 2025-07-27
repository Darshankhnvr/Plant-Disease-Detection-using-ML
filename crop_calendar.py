from datetime import datetime, timedelta
import json
import sqlite3

class CropCalendar:
    """
    Smart crop calendar with disease prevention alerts and treatment scheduling
    """
    
    def __init__(self, db_path='disease_tracking.db'):
        self.db_path = db_path
        self.init_calendar_db()
        
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
            harvest_days = sum(stage['days'] for stage in self.crop_stages.get(crop_type, {}).values())
            planting_dt = datetime.strptime(planting_date, '%Y-%m-%d')
            expected_harvest = (planting_dt + timedelta(days=harvest_days)).strftime('%Y-%m-%d')
            
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
            self.generate_crop_alerts_with_connection(cursor, calendar_id, crop_type, planting_dt)
            
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
    
    def generate_crop_alerts_with_connection(self, cursor, calendar_id, crop_type, planting_date):
        """Generate preventive alerts for crop stages using existing connection"""
        current_date = planting_date
        stages = self.crop_stages.get(crop_type, {})
        
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
                crop_id, user_id, crop_type, planting_date, location, variety, harvest_date, stage, created = crop
                
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
                    'harvest_date': harvest_date,
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