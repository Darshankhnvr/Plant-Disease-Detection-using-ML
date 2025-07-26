import sqlite3
import json
from datetime import datetime
import uuid

class DiseaseTracker:
    def __init__(self, db_path='disease_tracking.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Disease cases table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS disease_cases (
                id TEXT PRIMARY KEY,
                plant_name TEXT NOT NULL,
                initial_disease TEXT NOT NULL,
                initial_confidence REAL NOT NULL,
                initial_severity TEXT NOT NULL,
                initial_health_score INTEGER NOT NULL,
                initial_image_path TEXT NOT NULL,
                created_date TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                notes TEXT
            )
        ''')
        
        # Disease progression table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS disease_progression (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                disease_detected TEXT NOT NULL,
                confidence REAL NOT NULL,
                severity TEXT NOT NULL,
                health_score INTEGER NOT NULL,
                analysis_date TEXT NOT NULL,
                treatment_applied TEXT,
                notes TEXT,
                FOREIGN KEY (case_id) REFERENCES disease_cases (id)
            )
        ''')
        
        # Treatment records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS treatments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                treatment_name TEXT NOT NULL,
                treatment_date TEXT NOT NULL,
                dosage TEXT,
                method TEXT,
                cost REAL,
                notes TEXT,
                FOREIGN KEY (case_id) REFERENCES disease_cases (id)
            )
        ''')
        
        # Yield predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS yield_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                predicted_yield_loss REAL NOT NULL,
                economic_impact REAL NOT NULL,
                prediction_date TEXT NOT NULL,
                harvest_recommendation TEXT,
                FOREIGN KEY (case_id) REFERENCES disease_cases (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_disease_case(self, plant_name, prediction_data, image_path, notes=""):
        """Create a new disease case"""
        case_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO disease_cases 
            (id, plant_name, initial_disease, initial_confidence, initial_severity, 
             initial_health_score, initial_image_path, created_date, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            case_id,
            plant_name,
            prediction_data['primary_disease']['name'],
            prediction_data['confidence'],
            prediction_data['severity'],
            prediction_data['health_score'],
            image_path,
            datetime.now().isoformat(),
            notes
        ))
        
        conn.commit()
        conn.close()
        return case_id
    
    def add_progression_entry(self, case_id, prediction_data, image_path, treatment_applied="", notes=""):
        """Add a new progression entry for existing case"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO disease_progression 
            (case_id, image_path, disease_detected, confidence, severity, 
             health_score, analysis_date, treatment_applied, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            case_id,
            image_path,
            prediction_data['primary_disease']['name'],
            prediction_data['confidence'],
            prediction_data['severity'],
            prediction_data['health_score'],
            datetime.now().isoformat(),
            treatment_applied,
            notes
        ))
        
        conn.commit()
        conn.close()
    
    def get_disease_case(self, case_id):
        """Get disease case details"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM disease_cases WHERE id = ?', (case_id,))
        case = cursor.fetchone()
        
        if case:
            cursor.execute('''
                SELECT * FROM disease_progression 
                WHERE case_id = ? 
                ORDER BY analysis_date ASC
            ''', (case_id,))
            progression = cursor.fetchall()
            
            conn.close()
            return {
                'case': {
                    'id': case[0],
                    'plant_name': case[1],
                    'initial_disease': case[2],
                    'initial_confidence': case[3],
                    'initial_severity': case[4],
                    'initial_health_score': case[5],
                    'initial_image_path': case[6],
                    'created_date': case[7],
                    'status': case[8],
                    'notes': case[9] if len(case) > 9 else ''
                },
                'progression': [
                    {
                        'id': p[0],
                        'case_id': p[1],
                        'image_path': p[2],
                        'disease_detected': p[3],
                        'confidence': p[4],
                        'severity': p[5],
                        'health_score': p[6],
                        'analysis_date': p[7],
                        'treatment_applied': p[8] if len(p) > 8 else '',
                        'notes': p[9] if len(p) > 9 else ''
                    } for p in progression
                ]
            }
        
        conn.close()
        return None
    
    def get_all_cases(self):
        """Get all disease cases"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, plant_name, initial_disease, initial_severity, 
                   created_date, status 
            FROM disease_cases 
            ORDER BY created_date DESC
        ''')
        cases = cursor.fetchall()
        
        conn.close()
        return cases
    
    def calculate_treatment_effectiveness(self, case_id):
        """Calculate treatment effectiveness based on progression"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get initial health score
        cursor.execute('SELECT initial_health_score FROM disease_cases WHERE id = ?', (case_id,))
        initial_health = cursor.fetchone()[0]
        
        # Get latest health score
        cursor.execute('''
            SELECT health_score FROM disease_progression 
            WHERE case_id = ? 
            ORDER BY analysis_date DESC 
            LIMIT 1
        ''', (case_id,))
        
        latest_result = cursor.fetchone()
        if latest_result:
            latest_health = latest_result[0]
            improvement = latest_health - initial_health
            effectiveness_percentage = (improvement / (100 - initial_health)) * 100 if initial_health < 100 else 0
            
            conn.close()
            return {
                'initial_health': initial_health,
                'current_health': latest_health,
                'improvement': improvement,
                'effectiveness_percentage': max(0, min(100, effectiveness_percentage))
            }
        
        conn.close()
        return None
    
    def add_treatment_record(self, case_id, treatment_name, dosage="", method="", cost=0.0, notes=""):
        """Add treatment record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO treatments 
            (case_id, treatment_name, treatment_date, dosage, method, cost, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            case_id,
            treatment_name,
            datetime.now().isoformat(),
            dosage,
            method,
            cost,
            notes
        ))
        
        conn.commit()
        conn.close()
    
    def get_treatments(self, case_id):
        """Get all treatments for a case"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM treatments 
            WHERE case_id = ? 
            ORDER BY treatment_date DESC
        ''', (case_id,))
        
        treatments = cursor.fetchall()
        conn.close()
        return treatments