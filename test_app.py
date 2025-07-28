#!/usr/bin/env python3
"""
Simple test script to identify the specific issue with the analyze plant disease functionality
"""

import sys
import traceback

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import flask
        print("✓ Flask imported successfully")
    except Exception as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print("✓ TensorFlow imported successfully")
    except Exception as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except Exception as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ PIL imported successfully")
    except Exception as e:
        print(f"✗ PIL import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if the model can be loaded"""
    print("\nTesting model loading...")
    
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")
        print("✓ Model loaded successfully")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        traceback.print_exc()
        return False

def test_enhanced_predictor():
    """Test if the enhanced predictor can be initialized"""
    print("\nTesting enhanced predictor...")
    
    try:
        from enhanced_model_predict import initialize_hybrid_predictor
        success = initialize_hybrid_predictor()
        if success:
            print("✓ Enhanced predictor initialized successfully")
            return True
        else:
            print("✗ Enhanced predictor initialization failed")
            return False
    except Exception as e:
        print(f"✗ Enhanced predictor test failed: {e}")
        traceback.print_exc()
        return False

def test_json_loading():
    """Test if plant_disease.json can be loaded"""
    print("\nTesting JSON loading...")
    
    try:
        import json
        with open("plant_disease.json", 'r') as file:
            plant_disease = json.load(file)
        print(f"✓ plant_disease.json loaded successfully ({len(plant_disease)} entries)")
        return True
    except Exception as e:
        print(f"✗ JSON loading failed: {e}")
        return False

def test_database_modules():
    """Test if database modules can be imported"""
    print("\nTesting database modules...")
    
    modules_to_test = [
        'database',
        'yield_predictor', 
        'environmental_correlator',
        'weather_integration',
        'crop_calendar',
        'smart_treatment_advisor',
        'analytics_dashboard'
    ]
    
    all_success = True
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name} imported successfully")
        except Exception as e:
            print(f"✗ {module_name} import failed: {e}")
            all_success = False
    
    return all_success

def main():
    """Run all tests"""
    print("=== CropSense App Diagnostic Test ===\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Model Loading Test", test_model_loading),
        ("Enhanced Predictor Test", test_enhanced_predictor),
        ("JSON Loading Test", test_json_loading),
        ("Database Modules Test", test_database_modules)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            traceback.print_exc()
            results[test_name] = False
        print()
    
    print("=== Test Summary ===")
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! The app should work correctly.")
        print("If you're still experiencing issues, the problem might be:")
        print("- Browser-related (try clearing cache/cookies)")
        print("- Network-related (check if Flask server is running)")
        print("- File permissions (check if uploadimages directory is writable)")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main()