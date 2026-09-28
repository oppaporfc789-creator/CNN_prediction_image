# CNN_prediction_image
That's a great project to show on your GitHub profile! Animal classification using CNNs and Keras shows solid computer vision and deep learning fundamentals.
# Multi-Class Animal Image Classifier (CNN)

An end-to-end Computer Vision pipeline built with **TensorFlow / Keras** and **OpenCV** to classify 5 animal species: **Cat, Cow, Dog, Elephant, and Sheep**.

## 📌 Features
- **Architecture:** Custom Convolutional Neural Network (CNN) with Dropout optimization to reduce overfitting.
- **Data Pipeline:** Standardized image preprocessing and categorical flow generators.
- **Inference:** Single-image and batch prediction testing scripts using `np.argmax`.

## 📊 Performance
- **Validation Accuracy:** ~60% (Custom CNN architecture baseline)
- **Loss Function:** Categorical Cross-Entropy

## 🛠️ Setup & Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/animal-cnn-classifier.git](https://github.com/your-username/animal-cnn-classifier.git)
   cd animal-cnn-classifier
   ```

2. Create a virtual environment & install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run prediction on test images:
   ```bash
   python predict.py
   ```
