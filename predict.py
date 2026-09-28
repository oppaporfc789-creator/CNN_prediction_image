import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Class mapping corresponding to categorical training index order
CLASS_NAMES = ['Cat 🐈', 'Cow 🐄', 'Dog 🐶', 'Elephant 🐘', 'Sheep 🐑']
MODEL_PATH = 'models/CNN_predict_model.h5'
TEST_DIR = 'Dataset/test'

def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Train the model first using train.py.")
    return load_model(MODEL_PATH)

def predict_single_image(model, img_path):
    """Load, preprocess, and predict a single image file."""
    img = image.load_img(img_path, target_size=(200, 200))
    
    # Display image
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    # Preprocess (array conversion & normalization)
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)  # Shape: (1, 200, 200, 3)
    
    # Run prediction
    preds = model.predict(x, verbose=0)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx] * 100
    
    print(f"File: {os.path.basename(img_path)}")
    print(f"Prediction: {CLASS_NAMES[class_idx]} ({confidence:.2f}% confidence)\n")

def run_batch_predictions(model, folder_path):
    """Recursively predict images inside test directory using os.walk."""
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                predict_single_image(model, img_path)

if __name__ == '__main__':
    cnn_model = load_trained_model()
    run_batch_predictions(cnn_model, TEST_DIR)