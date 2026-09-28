import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import CSVLogger

# 1. Hyperparameters & Configuration
IMG_SIZE = (200, 200)
BATCH_SIZE = 10
EPOCHS = 10
CLASSES = ['Cat', 'Cow', 'Dog', 'Elephant', 'Sheep']
NUM_CLASSES = len(CLASSES)

TRAIN_DIR = 'Dataset/train'
VAL_DIR = 'Dataset/validation'
MODEL_DIR = 'models'

os.makedirs(MODEL_DIR, exist_ok=True)

# 2. Data Generators (Pixel Rescaling)
train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_dataset = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    class_mode='categorical'
)

val_dataset = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    class_mode='categorical'
)

# 3. Model Architecture Construction
model = Sequential([
    Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPool2D(2, 2),
    
    Conv2D(32, (3, 3), activation='relu'),
    MaxPool2D(2, 2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Reduces overfitting
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Model Training
print("Starting Model Training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[CSVLogger('history.csv')],
    shuffle=True
)

# 5. Evaluate Real Validation Performance
val_loss, val_acc = model.evaluate(val_dataset)
print(f"\nFinal Validation Accuracy: {round(val_acc * 100, 2)}%")
print(f"Final Validation Loss: {round(val_loss, 4)}")

# 6. Save Trained Model
model_save_path = os.path.join(MODEL_DIR, 'CNN_predict_model.h5')
model.save(model_save_path)
print(f"Model successfully saved to '{model_save_path}'.")