import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam  # Import optimizer
from preprocessing_valid import get_validation_generator  # Import validation generator

# Paths to validation images and labels
val_images_path = "../data/brain-tumor/valid/images"
val_labels_path = "../data/brain-tumor/valid/labels"

# Model path
MODEL_PATH = "brain_tumor_classifier.h5"  # Update if the model filename differs

# Load trained model
model = load_model(MODEL_PATH, compile=False)

# Compile the model (this is necessary before evaluation)
model.compile(optimizer=Adam(learning_rate=0.0001),  # Adjust learning rate if needed
              loss="binary_crossentropy", 
              metrics=["accuracy"])

# Create validation generator
val_generator = get_validation_generator(val_images_path, val_labels_path)

# Evaluate model on validation set
val_loss, val_acc = model.evaluate(val_generator, verbose=1)

print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")