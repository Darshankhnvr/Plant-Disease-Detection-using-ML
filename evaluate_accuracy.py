import tensorflow as tf
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and labels
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")

label = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

with open("plant_disease.json",'r') as file:
    plant_disease = json.load(file)

def extract_features(image_path):
    """Extract features from image for prediction"""
    image = tf.keras.utils.load_img(image_path, target_size=(160,160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

def predict_image(image_path):
    """Predict disease for a single image"""
    img = extract_features(image_path)
    prediction = model.predict(img, verbose=0)
    predicted_class = prediction.argmax()
    confidence = prediction.max()
    return predicted_class, confidence, label[predicted_class]

def evaluate_model_summary():
    """Display model summary and basic information"""
    print("=" * 60)
    print("PLANT DISEASE RECOGNITION MODEL EVALUATION")
    print("=" * 60)
    
    print(f"Model Architecture:")
    model.summary()
    
    print(f"\nNumber of classes: {len(label)}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
    return len(label)

def evaluate_uploaded_images():
    """Evaluate model on uploaded images"""
    upload_dir = "uploadimages"
    
    if not os.path.exists(upload_dir):
        print(f"Upload directory '{upload_dir}' not found!")
        return
    
    image_files = [f for f in os.listdir(upload_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("No image files found in uploadimages directory!")
        return
    
    print(f"\nEvaluating {len(image_files)} uploaded images...")
    print("-" * 60)
    
    predictions = []
    confidences = []
    
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(upload_dir, image_file)
        try:
            predicted_class, confidence, predicted_label = predict_image(image_path)
            predictions.append({
                'file': image_file,
                'predicted_class': predicted_class,
                'predicted_label': predicted_label,
                'confidence': confidence
            })
            confidences.append(confidence)
            
            print(f"{i:2d}. {image_file[:50]:<50} -> {predicted_label[:30]:<30} ({confidence:.3f})")
            
        except Exception as e:
            print(f"{i:2d}. {image_file[:50]:<50} -> ERROR: {str(e)}")
    
    if confidences:
        print(f"\nConfidence Statistics:")
        print(f"Average confidence: {np.mean(confidences):.3f}")
        print(f"Min confidence: {np.min(confidences):.3f}")
        print(f"Max confidence: {np.max(confidences):.3f}")
        print(f"Std deviation: {np.std(confidences):.3f}")
        
        # Distribution of predictions
        class_counts = {}
        for pred in predictions:
            label_name = pred['predicted_label']
            class_counts[label_name] = class_counts.get(label_name, 0) + 1
        
        print(f"\nPrediction Distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count} images")
    
    return predictions

def analyze_model_performance():
    """Analyze model performance metrics"""
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Get model compile metrics if available
    try:
        if hasattr(model, 'compiled_metrics'):
            print("Compiled metrics:", model.compiled_metrics.metrics)
        if hasattr(model, 'optimizer'):
            print("Optimizer:", model.optimizer)
    except:
        pass
    
    # Analyze model layers
    print(f"\nModel Layers Analysis:")
    total_params = 0
    trainable_params = 0
    
    for i, layer in enumerate(model.layers):
        layer_params = layer.count_params()
        total_params += layer_params
        if layer.trainable:
            trainable_params += layer_params
        
        if i < 10:  # Show first 10 layers
            print(f"  {i+1:2d}. {layer.name:<20} {str(layer.__class__.__name__):<15} Params: {layer_params:>8,}")
    
    if len(model.layers) > 10:
        print(f"  ... and {len(model.layers) - 10} more layers")
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

def main():
    """Main evaluation function"""
    print("Starting Plant Disease Model Evaluation...\n")
    
    # Model summary
    num_classes = evaluate_model_summary()
    
    # Analyze model performance
    analyze_model_performance()
    
    # Evaluate on uploaded images
    predictions = evaluate_uploaded_images()
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    if predictions:
        high_confidence = [p for p in predictions if p['confidence'] > 0.9]
        medium_confidence = [p for p in predictions if 0.7 <= p['confidence'] <= 0.9]
        low_confidence = [p for p in predictions if p['confidence'] < 0.7]
        
        print(f"\nConfidence Distribution:")
        print(f"High confidence (>0.9): {len(high_confidence)} images")
        print(f"Medium confidence (0.7-0.9): {len(medium_confidence)} images")
        print(f"Low confidence (<0.7): {len(low_confidence)} images")
        
        if low_confidence:
            print(f"\nLow confidence predictions (may need review):")
            for pred in low_confidence:
                print(f"  {pred['file'][:40]:<40} -> {pred['predicted_label'][:25]:<25} ({pred['confidence']:.3f})")

if __name__ == "__main__":
    main()