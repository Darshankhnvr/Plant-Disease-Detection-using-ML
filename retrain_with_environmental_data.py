import tensorflow as tf
import numpy as np
import os
from hybrid_disease_model import HybridDiseaseModel
import json

def load_existing_model_data():
    """
    Load your existing model and extract training data
    This is a simulation - you'd need your actual training dataset
    """
    
    # Load your existing model to understand its structure
    try:
        existing_model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")
        print("Existing model loaded successfully")
        print(f"Model input shape: {existing_model.input_shape}")
        print(f"Model output shape: {existing_model.output_shape}")
        return existing_model
    except Exception as e:
        print(f"Could not load existing model: {e}")
        return None

def create_hybrid_training_data(num_samples_per_class=100):
    """
    Create synthetic training data combining images and environmental data
    In practice, you'd use your actual image dataset
    """
    
    # Your disease labels
    labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
              'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Septoria_leaf_spot', 'Tomato___healthy',
              'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
              'Corn___Common_rust', 'Corn___healthy', 'Grape___Black_rot', 'Grape___healthy']
    
    # Create synthetic data
    all_images = []
    all_env_data = []
    all_labels = []
    
    hybrid_model = HybridDiseaseModel()
    
    for i, label in enumerate(labels):
        for _ in range(num_samples_per_class):
            # Create synthetic image (in practice, use your real images)
            synthetic_image = np.random.rand(160, 160, 3)
            all_images.append(synthetic_image)
            
            # Create label
            label_vector = np.zeros(len(labels))
            label_vector[i] = 1
            all_labels.append(label_vector)
        
        # Create synthetic environmental data for this disease
        env_data_for_disease = hybrid_model.create_synthetic_environmental_data(
            [label] * num_samples_per_class, num_samples_per_class
        )
        all_env_data.extend(env_data_for_disease)
    
    return np.array(all_images), np.array(all_env_data), np.array(all_labels)

def train_hybrid_model():
    """
    Train the hybrid model with both image and environmental data
    """
    
    print("Creating hybrid model...")
    hybrid_model = HybridDiseaseModel(num_classes=15)  # Adjust based on your classes
    hybrid_model.create_hybrid_model()
    hybrid_model.compile_model()
    
    print("\nModel Architecture:")
    hybrid_model.get_model_summary()
    
    print("\nGenerating synthetic training data...")
    images, env_data, labels = create_hybrid_training_data(num_samples_per_class=50)
    
    print(f"Training data shapes:")
    print(f"Images: {images.shape}")
    print(f"Environmental data: {env_data.shape}")
    print(f"Labels: {labels.shape}")
    
    # Normalize environmental data
    env_data_normalized = hybrid_model.prepare_environmental_data(env_data)
    
    # Split data
    split_idx = int(0.8 * len(images))
    
    train_images = images[:split_idx]
    train_env = env_data_normalized[:split_idx]
    train_labels = labels[:split_idx]
    
    val_images = images[split_idx:]
    val_env = env_data_normalized[split_idx:]
    val_labels = labels[split_idx:]
    
    print(f"\nTraining set: {len(train_images)} samples")
    print(f"Validation set: {len(val_images)} samples")
    
    # Train the model
    print("\nStarting training...")
    history = hybrid_model.model.fit(
        [train_images, train_env],
        train_labels,
        validation_data=([val_images, val_env], val_labels),
        epochs=10,  # Increase for real training
        batch_size=32,
        verbose=1
    )
    
    # Save the trained model
    os.makedirs('models/hybrid', exist_ok=True)
    hybrid_model.save_model(
        'models/hybrid/hybrid_disease_model.keras',
        'models/hybrid/env_scaler.pkl'
    )
    
    print("\nHybrid model training completed!")
    print("Model saved to: models/hybrid/")
    
    return hybrid_model, history

def test_hybrid_prediction():
    """
    Test the hybrid model with sample data
    """
    
    # Load the trained model
    hybrid_model = HybridDiseaseModel()
    try:
        hybrid_model.load_model(
            'models/hybrid/hybrid_disease_model.keras',
            'models/hybrid/env_scaler.pkl'
        )
        print("Hybrid model loaded successfully!")
        
        # Test with sample data
        test_image = np.random.rand(160, 160, 3)  # Replace with real image
        test_env_data = [6.5, 25.0, 80.0]  # pH, temperature, humidity
        
        prediction = hybrid_model.predict_hybrid(test_image, test_env_data)
        
        print(f"\nTest prediction shape: {prediction.shape}")
        print(f"Predicted class: {np.argmax(prediction)}")
        print(f"Confidence: {np.max(prediction):.2%}")
        
    except Exception as e:
        print(f"Could not load hybrid model: {e}")

if __name__ == "__main__":
    print("=== Hybrid Disease Model Training ===")
    
    # Check existing model
    existing_model = load_existing_model_data()
    
    # Train hybrid model
    hybrid_model, history = train_hybrid_model()
    
    # Test the model
    test_hybrid_prediction()
    
    print("\n=== Training Complete ===")
    print("Next steps:")
    print("1. Replace synthetic images with your actual dataset")
    print("2. Collect real environmental data for training")
    print("3. Fine-tune the model architecture")
    print("4. Integrate the hybrid model into your Flask app")