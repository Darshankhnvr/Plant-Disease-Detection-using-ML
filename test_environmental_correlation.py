#!/usr/bin/env python3
"""
Test script to demonstrate environmental correlation functionality
"""

from environmental_correlator import EnvironmentalCorrelator

def test_environmental_correlation():
    correlator = EnvironmentalCorrelator()
    
    # Test case 1: High-risk conditions for Tomato Late Blight
    print("=== Test Case 1: Tomato Late Blight - High Risk Conditions ===")
    environmental_data_high_risk = {
        'temperature': 20.0,  # Perfect for late blight
        'humidity': 95.0,     # Very high humidity - perfect for disease
        'soil_ph': 6.2        # Optimal pH for disease
    }
    
    prediction_data = {
        'primary_disease': {'name': 'Tomato___Late_blight'},
        'confidence': 75.0,
        'severity': 'Moderate',
        'health_score': 60
    }
    
    enhanced_prediction = correlator.enhance_prediction_with_environment(
        prediction_data.copy(), environmental_data_high_risk
    )
    
    print(f"Original Confidence: {prediction_data['confidence']}%")
    print(f"Enhanced Confidence: {enhanced_prediction['confidence']}%")
    print(f"Environmental Risk: {enhanced_prediction['environmental_analysis']['environmental_risk']}")
    print(f"Risk Factors:")
    for factor in enhanced_prediction['environmental_analysis']['risk_factors']:
        print(f"  - {factor}")
    print()
    
    # Test case 2: Low-risk conditions for same disease
    print("=== Test Case 2: Tomato Late Blight - Low Risk Conditions ===")
    environmental_data_low_risk = {
        'temperature': 30.0,  # Too hot for late blight
        'humidity': 50.0,     # Too dry for disease
        'soil_ph': 7.5        # Not optimal pH
    }
    
    enhanced_prediction_low = correlator.enhance_prediction_with_environment(
        prediction_data.copy(), environmental_data_low_risk
    )
    
    print(f"Original Confidence: {prediction_data['confidence']}%")
    print(f"Enhanced Confidence: {enhanced_prediction_low['confidence']}%")
    print(f"Environmental Risk: {enhanced_prediction_low['environmental_analysis']['environmental_risk']}")
    print(f"Risk Factors:")
    for factor in enhanced_prediction_low['environmental_analysis']['risk_factors']:
        print(f"  - {factor}")
    print()
    
    # Test case 3: Environmental recommendations
    print("=== Test Case 3: Environmental Recommendations for Tomato ===")
    recommendations = correlator.get_environmental_recommendations('Tomato', environmental_data_high_risk)
    print("Recommendations:")
    for rec in recommendations['recommendations']:
        print(f"  - {rec}")
    print(f"Optimal Conditions: {recommendations['optimal_conditions']}")

if __name__ == "__main__":
    test_environmental_correlation()