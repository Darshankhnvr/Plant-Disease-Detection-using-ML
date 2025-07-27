#!/usr/bin/env python3
"""
Test the treatment advisor backend functionality
"""

from smart_treatment_advisor import SmartTreatmentAdvisor

def test_treatment_advisor():
    print("=== Testing Treatment Advisor Backend ===")
    
    try:
        advisor = SmartTreatmentAdvisor()
        print("✓ SmartTreatmentAdvisor initialized successfully")
        
        # Test with sample data
        test_data = {
            'disease_name': 'Tomato___Late_blight',
            'severity': 'Moderate',
            'environmental_data': {
                'soil_ph': 6.5,
                'temperature': 22.0,
                'humidity': 85.0
            },
            'crop_stage': 'vegetative',
            'farmer_preference': 'balanced',
            'budget_limit': None,
            'organic_only': False
        }
        
        print(f"\nTesting with: {test_data['disease_name']}, {test_data['severity']} severity")
        
        recommendations = advisor.get_smart_recommendations(
            disease_name=test_data['disease_name'],
            severity=test_data['severity'],
            environmental_data=test_data['environmental_data'],
            crop_stage=test_data['crop_stage'],
            farmer_preference=test_data['farmer_preference'],
            budget_limit=test_data['budget_limit'],
            organic_only=test_data['organic_only']
        )
        
        print("✓ Recommendations generated successfully!")
        print(f"  Primary recommendations: {len(recommendations.get('primary_recommendations', []))}")
        print(f"  IPM advice categories: {len(recommendations.get('imp_advice', {}))}")
        print(f"  Application schedule: {'Yes' if recommendations.get('application_schedule') else 'No'}")
        print(f"  Resistance warnings: {len(recommendations.get('resistance_warnings', []))}")
        
        # Print sample recommendation
        if recommendations.get('primary_recommendations'):
            first_rec = recommendations['primary_recommendations'][0]
            print(f"\nSample recommendation:")
            print(f"  Name: {first_rec.get('name', 'N/A')}")
            print(f"  Type: {first_rec.get('treatment_type', 'N/A')}")
            print(f"  Effectiveness: {first_rec.get('effectiveness', 'N/A')}%")
            print(f"  Cost: ${first_rec.get('cost_per_hectare', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing treatment advisor: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_treatment_advisor()
    if success:
        print("\n✓ Treatment Advisor backend is working correctly!")
        print("The frontend JavaScript should now work with the fixed code.")
    else:
        print("\n✗ Treatment Advisor backend has issues that need to be fixed.")