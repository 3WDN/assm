import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import (Input, TimeDistributed, Conv2D, MaxPooling2D, 
                                     Flatten, LSTM, Dense, Dropout)

def load_data(data_dir, time_steps=10, img_size=(64, 64)):
    X = []
    y = []

    # Updated labels to match your dataset structure
    labels = {
        'yawn': 0,
        'no_yawn': 1,
        'Open': 2,
        'Closed': 3
    }

    print(f"\nChecking directory: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return np.array([]), np.array([])

    for label in labels:
        label_folder = os.path.join(data_dir, label)
        if not os.path.exists(label_folder):
            print(f"Label folder not found: {label_folder}")
            continue
            
        print(f"\nProcessing {label} folder...")
        frames = sorted([f for f in os.listdir(label_folder) if f.endswith('.jpg')])
        print(f"Found {len(frames)} images")
        
        # Process images in batches of time_steps
        for i in range(0, len(frames) - time_steps + 1, time_steps):
            sequence = []
            for frame in frames[i:i + time_steps]:
                img_path = os.path.join(label_folder, frame)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to read image: {img_path}")
                    continue
                img = cv2.resize(img, img_size)
                img = img / 255.0
                sequence.append(img)

            if len(sequence) == time_steps:
                X.append(sequence)
                y.append(labels[label])

    print(f"\nTotal sequences loaded: {len(X)}")
    return np.array(X), np.array(y)

# Set dataset paths
train_path = "train"
test_path = "test"

# Load training data
print("\nLoading training data...")
X_train, y_train = load_data(train_path, time_steps=10, img_size=(64, 64))
print("Training data loaded.")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Load test data
print("\nLoading test data...")
X_test, y_test = load_data(test_path, time_steps=10, img_size=(64, 64))
print("Test data loaded.")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Check if we have any data
if len(X_train) == 0 or len(X_test) == 0:
    print("\nError: No data was loaded. Please check your dataset structure.")
    print("Current working directory:", os.getcwd())
    print("Expected structure:")
    print("train/")
    print("  ├── yawn/")
    print("  ├── no_yawn/")
    print("  ├── Open/")
    print("  └── Closed/")
    print("test/")
    print("  ├── yawn/")
    print("  ├── no_yawn/")
    print("  ├── Open/")
    print("  └── Closed/")
    exit(1)

# Convert labels to categorical
y_train_cat = to_categorical(y_train, 4)  # 4 classes
y_test_cat = to_categorical(y_test, 4)    # 4 classes

# Build CNN-LSTM Model
def build_cnn_lstm_model(input_shape=(10, 64, 64, 3), num_classes=4):
    input_layer = Input(shape=input_shape)
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(input_layer)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(128)(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output)
    return model

# Create and compile model
model = build_cnn_lstm_model(input_shape=(10, 64, 64, 3), num_classes=4)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=20,
    batch_size=16,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)

# Plot accuracy and loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat)
print(f"\nTest accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Save the model
model.save('drowsiness_detection_model.h5')
print("\nModel saved as 'drowsiness_detection_model.h5'")
