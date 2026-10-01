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

## 🛠️  Check below python code 
   ```bash
   model_fit = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20,
    callbacks=[tf.keras.callbacks.CSVLogger('history.csv')],
    shuffle=True
   )
   
   # Pass validation_dataset instead of train_dataset
   _, accuracy = model.evaluate(validation_dataset)
   print(f"Real Validation Accuracy: {round(accuracy * 100, 2)}%")
   ```


   ```bash
   epoch	accuracy	loss	val_accuracy	val_loss
   0	0	0.687778	0.864271	0.88	0.313227
   1	1	0.898889	0.304998	0.92	0.235467
   2	2	0.938889	0.219869	0.94	0.191619
   3	3	0.944444	0.190758	0.96	0.164977
   4	4	0.955556	0.140621	0.96	0.125185
   5	5	0.952222	0.140042	0.94	0.173105
   6	6	0.951111	0.124402	0.94	0.126470
   7	7	0.966667	0.120435	0.96	0.129652
   8	8	0.955556	0.132979	0.94	0.133552
   9	9	0.973333	0.089968	0.94	0.109342
   ```


   ```bash
      Model: "sequential_8"
   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
   ┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
   ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
   │ mobilenetv2_1.00_224            │ (None, 7, 7, 1280)     │     2,257,984 │
   │ (Functional)                    │                        │               │
   ├─────────────────────────────────┼────────────────────────┼───────────────┤
   │ global_average_pooling2d        │ (None, 1280)           │             0 │
   │ (GlobalAveragePooling2D)        │                        │               │
   ├─────────────────────────────────┼────────────────────────┼───────────────┤
   │ dropout_4 (Dropout)             │ (None, 1280)           │             0 │
   ├─────────────────────────────────┼────────────────────────┼───────────────┤
   │ dense_16 (Dense)                │ (None, 5)              │         6,405 │
   └─────────────────────────────────┴────────────────────────┴───────────────┘
    Total params: 2,277,201 (8.69 MB)
    Trainable params: 6,405 (25.02 KB)
    Non-trainable params: 2,257,984 (8.61 MB)
    Optimizer params: 12,812 (50.05 KB)
   ```

