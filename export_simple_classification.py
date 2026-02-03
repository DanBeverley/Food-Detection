#!/usr/bin/env python3
import os
import tensorflow as tf
import numpy as np
from pathlib import Path

def export_without_preprocessing():
    """Export TFLite model by removing preprocessing layer."""
    
    # Load the Keras model with custom objects
    custom_objects = {'preprocess_input': tf.keras.applications.efficientnet_v2.preprocess_input}
    
    try:
        model = tf.keras.models.load_model(
            "trained_models/classification/best_classification_model.keras",
            custom_objects=custom_objects,
            compile=False
        )
        print(f"Original model input shape: {model.input_shape}")
        print(f"Original model output shape: {model.output_shape}")
        
        # Get model without the first layer (preprocessing)
        if len(model.layers) > 1 and 'preprocessing' in model.layers[0].name:
            print("Removing preprocessing layer...")
            new_input = tf.keras.Input(shape=model.input_shape[1:], name='input')
            x = new_input
            for layer in model.layers[1:]:  # Skip first preprocessing layer
                x = layer(x)
            new_model = tf.keras.Model(inputs=new_input, outputs=x)
        else:
            new_model = model
            
        print(f"New model input shape: {new_model.input_shape}")
        print(f"New model output shape: {new_model.output_shape}")
        
        # Convert to TFLite without quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
        converter.optimizations = []  # No quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        
        tflite_model = converter.convert()
        
        # Save the model
        output_path = "trained_models/classification/exported/classification_model_final.tflite"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"Model exported to: {output_path}")
        print(f"Model size: {len(tflite_model) / (1024*1024):.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"Failed to export model: {e}")
        return False

if __name__ == "__main__":
    export_without_preprocessing()