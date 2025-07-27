import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

class AnalyticsDashboard:
    """
    Advanced analytics and insights dashboard for farm management
    """
    
    def __init__(self, db_path='disease_tracking.db'):
        self.db_path = db_path
    
    def get_disease_trends(self, time_period='30_days', crop_type=None):
        """Analyze disease trends over time"""
        conn = sqlite3.connect(self.db_path)
        
        # Calculate date range
        end_date = datetime.now()
        if time_period == '30_days':
            start_date = end_date - timedelta(days=30)
        elif time_period == '90_days':
            start_date = end_date - timedelta(days=90)
        elif time_period == '1_year':
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=30)
        
        # Query disease cases
        query = '''
            SELECT initial_disease, created_date, initial_severity, initial_confidence
            FROM disease_cases 
            WHERE created_date >= ? AND created_date <= ?
        '''
        params = [start_date.isoformat(), end_date.isoformat()]
        
        if crop_type:
            query += ' AND initial_disease LIKE ?'
            params.append(f'{crop_type}%')
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            return {'trends': [], 'summary': {}}
        
        # Process trends
        df['date'] = pd.to_datetime(df['created_date']).dt.date
        df['crop'] = df['initial_disease'].str.split('___').str[0]
        df['disease'] = df['initial_disease'].str.split('___').str[1]
        
        # Daily disease counts
        daily_trends = df.groupby(['date', 'disease']).size().reset_index(name='count')
        
        # Severity distribution
        severity_dist = df['initial_severity'].value_counts().to_dict()
        
        # Most affected crops
        crop_affected = df['crop'].value_counts().head(5).to_dict()
        
        # Confidence analysis
        avg_confidence = df['initial_confidence'].mean()
        
        trends_data = []
        for _, row in daily_trends.iterrows():
            trends_data.append({
                'date': row['date'].isoformat(),
                'disease': row['disease'],
                'count': int(row['count'])
            })
        
        return {
            'trends': trends_data,
            'summary': {
                'total_cases': len(df),
                'avg_confidence': round(avg_confidence, 1),
                'severity_distribution': severity_dist,
                'most_affected_crops': crop_affected,
                'period': time_period
            }
        }
    
    def get_treatment_effectiveness(self):
        """Analyze treatment effectiveness across all cases"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                dc.id,
                dc.initial_health_score,
                dc.initial_disease,
                dc.initial_severity,
                t.treatment_name,
                t.cost,
                dp.health_score as final_health_score,
                dp.analysis_date
            FROM disease_cases dc
            LEFT JOIN treatments t ON dc.id = t.case_id
            LEFT JOIN disease_progression dp ON dc.id = dp.case_id
            WHERE dp.health_score IS NOT NULL
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return {'effectiveness': [], 'summary': {}}
        
        # Calculate improvement
        df['health_improvement'] = df['final_health_score'] - df['initial_health_score']
        df['improvement_percentage'] = (df['health_improvement'] / (100 - df['initial_health_score'])) * 100
        
        # Treatment effectiveness by type
        treatment_effectiveness = df.groupby('treatment_name').agg({
            'health_improvement': 'mean',
            'improvement_percentage': 'mean',
            'cost': 'mean'
        }).round(2).to_dict('index')
        
        # Cost-effectiveness analysis
        df['cost_per_improvement'] = df['cost'] / (df['health_improvement'] + 1)  # +1 to avoid division by zero
        
        return {
            'effectiveness': [
                {
                    'treatment': treatment,
                    'avg_improvement': data['health_improvement'],
                    'improvement_percentage': data['improvement_percentage'],
                    'avg_cost': data['cost']
                }
                for treatment, data in treatment_effectiveness.items()
            ],
            'summary': {
                'total_treatments': len(df),
                'avg_improvement': round(df['health_improvement'].mean(), 1),
                'most_effective': df.loc[df['improvement_percentage'].idxmax(), 'treatment_name'] if not df.empty else None,
                'most_cost_effective': df.loc[df['cost_per_improvement'].idxmin(), 'treatment_name'] if not df.empty else None
            }
        }
    
    def get_environmental_correlations(self):
        """Analyze correlations between environmental factors and disease occurrence"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            query = '''
                SELECT 
                    initial_disease,
                    initial_severity,
                    initial_confidence,
                    environmental_data,
                    environmental_risk_score,
                    confidence_adjustment
                FROM disease_cases 
                WHERE environmental_data IS NOT NULL AND environmental_data != ''
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return {'correlations': [], 'insights': ['No environmental data available yet'], 'sample_size': 0}
            
            # Parse environmental data
            env_data_list = []
            for _, row in df.iterrows():
                try:
                    env_data = json.loads(row['environmental_data'])
                    if env_data:  # Only add if not empty
                        env_data['disease'] = row['initial_disease']
                        env_data['severity'] = row['initial_severity']
                        env_data['confidence'] = row['initial_confidence']
                        env_data['risk_score'] = row['environmental_risk_score']
                        env_data_list.append(env_data)
                except (json.JSONDecodeError, TypeError):
                    continue
            
            if not env_data_list:
                return {'correlations': [], 'insights': ['Environmental data exists but could not be parsed'], 'sample_size': 0}
            
            env_df = pd.DataFrame(env_data_list)
            
            # Calculate correlations
            correlations = []
            insights = []
            
            # Temperature correlations
            if 'temperature' in env_df.columns and not env_df['temperature'].isna().all():
                temp_disease_corr = env_df.groupby('disease')['temperature'].mean().to_dict()
                correlations.append({
                    'factor': 'temperature',
                    'disease_correlations': temp_disease_corr
                })
                
                # Find temperature insights
                high_temp_diseases = [disease for disease, temp in temp_disease_corr.items() if temp > 25]
                if high_temp_diseases:
                    insights.append(f"Diseases more common in high temperatures: {', '.join(high_temp_diseases[:3])}")
            
            # Humidity correlations
            if 'humidity' in env_df.columns and not env_df['humidity'].isna().all():
                humidity_disease_corr = env_df.groupby('disease')['humidity'].mean().to_dict()
                correlations.append({
                    'factor': 'humidity',
                    'disease_correlations': humidity_disease_corr
                })
                
                high_humidity_diseases = [disease for disease, humidity in humidity_disease_corr.items() if humidity > 80]
                if high_humidity_diseases:
                    insights.append(f"Diseases more common in high humidity: {', '.join(high_humidity_diseases[:3])}")
            
            # pH correlations
            if 'soil_ph' in env_df.columns and not env_df['soil_ph'].isna().all():
                ph_disease_corr = env_df.groupby('disease')['soil_ph'].mean().to_dict()
                correlations.append({
                    'factor': 'soil_ph',
                    'disease_correlations': ph_disease_corr
                })
            
            if not insights:
                insights.append('Environmental data collected but no significant patterns detected yet')
            
            return {
                'correlations': correlations,
                'insights': insights,
                'sample_size': len(env_df)
            }
            
        except Exception as e:
            conn.close()
            return {
                'correlations': [], 
                'insights': [f'Environmental analysis temporarily unavailable: {str(e)}'], 
                'sample_size': 0
            }
    
    def get_yield_impact_analysis(self):
        """Analyze yield impact across different scenarios"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                yp.predicted_yield_loss,
                yp.economic_impact,
                dc.initial_disease,
                dc.initial_severity,
                dc.initial_health_score
            FROM yield_predictions yp
            JOIN disease_cases dc ON yp.case_id = dc.id
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return {'impact_analysis': [], 'summary': {}}
        
        # Analyze by disease
        disease_impact = df.groupby('initial_disease').agg({
            'predicted_yield_loss': 'mean',
            'economic_impact': 'mean'
        }).round(2).to_dict('index')
        
        # Analyze by severity
        severity_impact = df.groupby('initial_severity').agg({
            'predicted_yield_loss': 'mean',
            'economic_impact': 'mean'
        }).round(2).to_dict('index')
        
        return {
            'impact_analysis': {
                'by_disease': disease_impact,
                'by_severity': severity_impact
            },
            'summary': {
                'total_cases': len(df),
                'avg_yield_loss': round(df['predicted_yield_loss'].mean(), 1),
                'total_economic_impact': round(df['economic_impact'].sum(), 2),
                'most_damaging_disease': df.loc[df['economic_impact'].idxmax(), 'initial_disease'] if not df.empty else None
            }
        }
    
    def get_seasonal_patterns(self):
        """Analyze seasonal disease patterns"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                initial_disease,
                created_date,
                initial_severity
            FROM disease_cases 
            WHERE created_date >= date('now', '-1 year')
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return {'patterns': [], 'insights': []}
        
        df['date'] = pd.to_datetime(df['created_date'])
        df['month'] = df['date'].dt.month
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Monthly patterns
        monthly_counts = df.groupby(['month', 'initial_disease']).size().reset_index(name='count')
        
        # Seasonal patterns
        seasonal_patterns = df.groupby(['season', 'initial_disease']).size().reset_index(name='count')
        
        # Generate insights
        insights = []
        peak_months = df.groupby('month').size()
        peak_month = peak_months.idxmax()
        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                      7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        
        insights.append(f"Peak disease activity occurs in {month_names[peak_month]}")
        
        # Most common diseases by season
        for season in ['Spring', 'Summer', 'Fall', 'Winter']:
            season_data = df[df['season'] == season]
            if not season_data.empty:
                common_disease = season_data['initial_disease'].value_counts().index[0]
                insights.append(f"Most common {season} disease: {common_disease.replace('___', ' ')}")
        
        return {
            'patterns': {
                'monthly': monthly_counts.to_dict('records'),
                'seasonal': seasonal_patterns.to_dict('records')
            },
            'insights': insights
        }
    
    def generate_farm_health_score(self, user_id='default'):
        """Generate overall farm health score"""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent cases (last 30 days)
        query = '''
            SELECT 
                initial_health_score,
                initial_severity,
                created_date
            FROM disease_cases 
            WHERE created_date >= date('now', '-30 days')
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return {
                'overall_score': 85,  # Default good score
                'status': 'Good',
                'factors': ['No recent disease cases detected']
            }
        
        # Calculate health score
        avg_health = df['initial_health_score'].mean()
        
        # Severity penalty
        severity_counts = df['initial_severity'].value_counts()
        severity_penalty = (
            severity_counts.get('Severe', 0) * 15 +
            severity_counts.get('Moderate', 0) * 8 +
            severity_counts.get('Mild', 0) * 3
        )
        
        # Recent activity penalty
        recent_cases = len(df)
        activity_penalty = min(recent_cases * 2, 20)
        
        # Calculate final score
        base_score = avg_health
        final_score = max(0, base_score - severity_penalty - activity_penalty)
        
        # Determine status
        if final_score >= 80:
            status = 'Excellent'
        elif final_score >= 65:
            status = 'Good'
        elif final_score >= 50:
            status = 'Fair'
        else:
            status = 'Poor'
        
        # Generate factors
        factors = []
        if recent_cases > 5:
            factors.append(f'{recent_cases} disease cases in last 30 days')
        if severity_counts.get('Severe', 0) > 0:
            factors.append(f'{severity_counts["Severe"]} severe cases detected')
        if avg_health < 70:
            factors.append('Below average plant health scores')
        
        if not factors:
            factors.append('Good disease management practices')
        
        return {
            'overall_score': round(final_score, 1),
            'status': status,
            'factors': factors,
            'recent_cases': recent_cases,
            'avg_plant_health': round(avg_health, 1)
        }

# Flask routes for analytics dashboard
def add_analytics_routes(app):
    """Add analytics routes to Flask app"""
    
    @app.route('/analytics-dashboard')
    def analytics_dashboard():
        analytics = AnalyticsDashboard()
        
        # Get all analytics data
        trends = analytics.get_disease_trends()
        effectiveness = analytics.get_treatment_effectiveness()
        correlations = analytics.get_environmental_correlations()
        yield_impact = analytics.get_yield_impact_analysis()
        seasonal = analytics.get_seasonal_patterns()
        farm_health = analytics.generate_farm_health_score()
        
        return render_template('analytics_dashboard.html',
                             trends=trends,
                             effectiveness=effectiveness,
                             correlations=correlations,
                             yield_impact=yield_impact,
                             seasonal=seasonal,
                             farm_health=farm_health)
    
    @app.route('/api/analytics/<analysis_type>')
    def get_analytics_data(analysis_type):
        analytics = AnalyticsDashboard()
        
        if analysis_type == 'trends':
            time_period = request.args.get('period', '30_days')
            crop_type = request.args.get('crop_type')
            return jsonify(analytics.get_disease_trends(time_period, crop_type))
        
        elif analysis_type == 'effectiveness':
            return jsonify(analytics.get_treatment_effectiveness())
        
        elif analysis_type == 'correlations':
            return jsonify(analytics.get_environmental_correlations())
        
        elif analysis_type == 'yield-impact':
            return jsonify(analytics.get_yield_impact_analysis())
        
        elif analysis_type == 'seasonal':
            return jsonify(analytics.get_seasonal_patterns())
        
        elif analysis_type == 'farm-health':
            return jsonify(analytics.generate_farm_health_score())
        
        else:
            return jsonify({'error': 'Invalid analysis type'}), 400