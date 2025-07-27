#!/usr/bin/env python3
"""
Test script to debug analytics dashboard issues
"""

from analytics_dashboard import AnalyticsDashboard
import traceback

def test_analytics():
    print("=== Testing Analytics Dashboard ===")
    
    try:
        analytics = AnalyticsDashboard()
        print("✓ AnalyticsDashboard initialized successfully")
        
        # Test each function individually
        print("\n1. Testing disease trends...")
        try:
            trends = analytics.get_disease_trends()
            print(f"✓ Disease trends: {len(trends.get('trends', []))} entries")
            print(f"  Summary: {trends.get('summary', {})}")
        except Exception as e:
            print(f"✗ Disease trends failed: {e}")
            traceback.print_exc()
        
        print("\n2. Testing treatment effectiveness...")
        try:
            effectiveness = analytics.get_treatment_effectiveness()
            print(f"✓ Treatment effectiveness: {len(effectiveness.get('effectiveness', []))} treatments")
            print(f"  Summary: {effectiveness.get('summary', {})}")
        except Exception as e:
            print(f"✗ Treatment effectiveness failed: {e}")
            traceback.print_exc()
        
        print("\n3. Testing environmental correlations...")
        try:
            correlations = analytics.get_environmental_correlations()
            print(f"✓ Environmental correlations: {len(correlations.get('correlations', []))} factors")
            print(f"  Insights: {len(correlations.get('insights', []))} insights")
        except Exception as e:
            print(f"✗ Environmental correlations failed: {e}")
            traceback.print_exc()
        
        print("\n4. Testing yield impact...")
        try:
            yield_impact = analytics.get_yield_impact_analysis()
            print(f"✓ Yield impact analysis completed")
            print(f"  Summary: {yield_impact.get('summary', {})}")
        except Exception as e:
            print(f"✗ Yield impact failed: {e}")
            traceback.print_exc()
        
        print("\n5. Testing seasonal patterns...")
        try:
            seasonal = analytics.get_seasonal_patterns()
            print(f"✓ Seasonal patterns: {len(seasonal.get('patterns', {}).get('monthly', []))} monthly entries")
            print(f"  Insights: {len(seasonal.get('insights', []))} insights")
        except Exception as e:
            print(f"✗ Seasonal patterns failed: {e}")
            traceback.print_exc()
        
        print("\n6. Testing farm health score...")
        try:
            farm_health = analytics.generate_farm_health_score()
            print(f"✓ Farm health score: {farm_health.get('overall_score', 'N/A')}")
            print(f"  Status: {farm_health.get('status', 'N/A')}")
        except Exception as e:
            print(f"✗ Farm health score failed: {e}")
            traceback.print_exc()
        
        print("\n=== Analytics Test Complete ===")
        
    except Exception as e:
        print(f"✗ Failed to initialize AnalyticsDashboard: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_analytics()