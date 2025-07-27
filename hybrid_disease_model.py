import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

class HybridDiseaseModel:
    """
    Hybrid model that combines image features with environmental data
    for more accurate disease prediction
    """
    
    def __init__(self, num_classes=39, image_size=(160, 160)):
        self.num_classes = num_classes
        self.image_size = image_size
        self.model = None
        self.env_scaler = StandardScaler()
        
    def create_hybrid_model(self):
        """
        Create a hybrid model with two input branches:
        1. Image branch (CNN)
        2. Environmental data branch (Dense layers)
        """
        
        # Image input branch
        image_input = keras.Input(shape=(*self.image_size, 3), name='image_input')
        
        # Use a pre-trained CNN backbone (MobileNetV2)
        base_model = keras.applications.MobileNetV2(
            input_shape=(*self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False  # Freeze base model initially
        
        # Image feature extraction
        x_img = base_model(image_input)
        x_img = layers.GlobalAveragePooling2D()(x_img)
        x_img = layers.Dropout(0.2)(x_img)
        image_features = layers.Dense(128, activation='relu', name='image_features')(x_img)
        
        # Environmental input branch
        env_input = keras.Input(shape=(3,), name='env_input')  # pH, temp, humidity
        x_env = layers.Dense(32, activation='relu')(env_input)
        x_env = layers.Dropout(0.2)(x_env)
        x_env = layers.Dense(16, activation='relu')(x_env)
        env_features = layers.Dense(8, activation='relu', name='env_features')(x_env)
        
        # Combine both branches
        combined = layers.concatenate([image_features, env_features])
        x = layers.Dense(64, activation='relu')(combined)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        # Create the model
        self.model = keras.Model(
            inputs=[image_input, env_input],
            outputs=outputs,
            name='hybrid_disease_model'
        )
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the hybrid model"""
        if self.model is None:
            self.create_hybrid_model()
            
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
    def prepare_environmental_data(self, env_data_list):
        """
        Prepare environmental data for training/prediction
        env_data_list: List of [pH, temperature, humidity] arrays
        """
        env_array = np.array(env_data_list)
        
        # Handle missing values (replace with median)
        for i in range(env_array.shape[1]):
            col = env_array[:, i]
            median_val = np.nanmedian(col[col != 0])  # Exclude zeros as missing
            env_array[:, i] = np.where((col == 0) | np.isnan(col), median_val, col)
        
        # Normalize environmental data
        env_normalized = self.env_scaler.fit_transform(env_array)
        return env_normalized
    
    def create_synthetic_environmental_data(self, disease_labels, num_samples):
        """
        Create synthetic environmental data based on disease-environment correlations
        This is needed since your original dataset doesn't have environmental data
        """
        
        # Disease-environment correlation patterns
        disease_env_patterns = {
            'Apple___Apple_scab': {'ph': (6.0, 7.0), 'temp': (18, 24), 'humidity': (85, 95)},
            'Apple___Black_rot': {'ph': (6.0, 7.5), 'temp': (24, 28), 'humidity': (75, 85)},
            'Tomato___Early_blight': {'ph': (6.0, 7.0), 'temp': (26, 28), 'humidity': (92, 98)},
            'Tomato___Late_blight': {'ph': (5.5, 7.0), 'temp': (18, 22), 'humidity': (90, 100)},
            'Potato___Late_blight': {'ph': (5.0, 6.5), 'temp': (18, 22), 'humidity': (90, 100)},
            'Corn___Common_rust': {'ph': (6.0, 7.5), 'temp': (22, 28), 'humidity': (80, 90)},
            'Grape___Black_rot': {'ph': (6.0, 7.0), 'temp': (24, 28), 'humidity': (80, 90)},
            # Add healthy plant patterns
            'Apple___healthy': {'ph': (6.0, 7.0), 'temp': (18, 24), 'humidity': (60, 75)},
            'Tomato___healthy': {'ph': (6.0, 6.8), 'temp': (20, 25), 'humidity': (65, 80)},
            'Potato___healthy': {'ph': (5.5, 6.5), 'temp': (15, 20), 'humidity': (70, 85)},
        }
        
        synthetic_env_data = []
        
        for label in disease_labels:
            if label in disease_env_patterns:
                pattern = disease_env_patterns[label]
                
                # Generate realistic environmental data with some noise
                ph = np.random.normal(
                    (pattern['ph'][0] + pattern['ph'][1]) / 2, 
                    (pattern['ph'][1] - pattern['ph'][0]) / 6
                )
                temp = np.random.normal(
                    (pattern['temp'][0] + pattern['temp'][1]) / 2,
                    (pattern['temp'][1] - pattern['temp'][0]) / 6
                )
                humidity = np.random.normal(
                    (pattern['humidity'][0] + pattern['humidity'][1]) / 2,
                    (pattern['humidity'][1] - pattern['humidity'][0]) / 6
                )
                
                # Clip to reasonable ranges
                ph = np.clip(ph, 4.0, 9.0)
                temp = np.clip(temp, 5.0, 45.0)
                humidity = np.clip(humidity, 20.0, 100.0)
                
            else:
                # Default values for unknown diseases
                ph = np.random.normal(6.5, 0.5)
                temp = np.random.normal(22, 5)
                humidity = np.random.normal(70, 15)
                
                ph = np.clip(ph, 4.0, 9.0)
                temp = np.clip(temp, 5.0, 45.0)
                humidity = np.clip(humidity, 20.0, 100.0)
            
            synthetic_env_data.append([ph, temp, humidity])
        
        return np.array(synthetic_env_data)
    
    def predict_hybrid(self, image, environmental_data):
        """
        Make prediction using both image and environmental data
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load or train the model first.")
        
        # Prepare image
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Prepare environmental data
        env_data = np.array(environmental_data).reshape(1, -1)
        env_data_normalized = self.env_scaler.transform(env_data)
        
        # Make prediction
        prediction = self.model.predict([image, env_data_normalized])
        return prediction
    
    def save_model(self, model_path, scaler_path):
        """Save the trained model and scaler"""
        if self.model is not None:
            self.model.save(model_path)
            
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.env_scaler, f)
    
    def load_model(self, model_path, scaler_path):
        """Load the trained model and scaler"""
        self.model = keras.models.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            self.env_scaler = pickle.load(f)
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is not None:
            return self.model.summary()
        else:
            return "Model not created yet."

# Example usage and training script
def create_training_pipeline():
    """
    Example of how to retrain your model with environmental data
    """
    
    # Initialize hybrid model
    hybrid_model = HybridDiseaseModel()
    hybrid_model.create_hybrid_model()
    hybrid_model.compile_model()
    
    print("Hybrid Model Architecture:")
    print(hybrid_model.get_model_summary())
    
    return hybrid_model

if __name__ == "__main__":
    model = create_training_pipeline()