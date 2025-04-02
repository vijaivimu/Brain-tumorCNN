import os
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from preprocessing_pretrain import get_train_generator

# Define dataset paths
IMAGE_DIR = "../data/brain-tumor/train/images"
LABEL_DIR = "../data/brain-tumor/train/labels"

# Get data generator
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

train_generator = get_train_generator(IMAGE_DIR, LABEL_DIR, batch_size=BATCH_SIZE, target_size=IMG_SIZE)

# Load pretrained EfficientNetB0 (without top layer)
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Unfreeze last few layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:150]:  # Freeze only first 150 layers
    layer.trainable = False

# Build custom classifier
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")  # Binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Train the model
EPOCHS = 10
model.fit(train_generator, epochs=EPOCHS, steps_per_epoch=len(train_generator), verbose=1)

# Save the model
model.save("brain_tumor_classifier.h5")
print("Model training complete. Saved as 'brain_tumor_classifier.h5'.")