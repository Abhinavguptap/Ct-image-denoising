"""
Image Denoising using DnCNN with TensorFlow and Keras
Introduction
This script implements a Deep Neural Network for image denoising using the DnCNN (Deep Convolutional Neural Network) architecture. The model is built using TensorFlow and Keras, and it is trained to remove noise from grayscale images.

Code Structure
Importing Libraries:
os, cv2, numpy, matplotlib.pyplot: Standard libraries for file operations, image processing, numerical operations, and plotting.
layers, models from tensorflow.keras: Components for building the neural network.

Data Loading and Preprocessing:
load_and_preprocess_data(data_dir, image_size): Function to load and preprocess images.
Noisy and clean images are loaded, resized to a specified image_size, normalized to the range [0, 1], and expanded with a channel dimension.

Loading Noisy and Clean Images:
Noisy and clean images are loaded using the load_and_preprocess_data function.

Building DnCNN Model:
build_dncnn_model(input_shape): Function to create the DnCNN model with convolutional layers.
The model architecture consists of two convolutional layers with ReLU activation and a final convolutional layer for output.

Compiling the Model:
The model is compiled using the Adam optimizer and Mean Squared Error (MSE) loss.

Model Training:
The model is trained using the fit function.
Training data includes noisy images, and target data includes corresponding clean images.
Training is done over a specified number of epochs with a batch size of 16 and a validation split of 20%.

Saving the Trained Model:
The trained model is saved in an h5 file (“denoising_model.h5”).

Plotting Training and Validation Loss Curves:
Matplotlib is used to display training and validation loss curves.

Denoising New Images:
A new noisy image is loaded and preprocessed.
The trained model is used to predict the denoised image.
Displaying Results:
Matplotlib is used to display the original noisy image, as well as the denoised image.

Conclusion:
The script provides a basic implementation of image denoising using a DnCNN model. Further experimentation and tuning may be required for specific use cases.
Usage
Ensure that the required libraries are installed (cv2, numpy, matplotlib, tensorflow).
Specify the paths to the noisy and clean image directories.
Adjust hyperparameters and model architecture as needed.
Run the script.
Recommendations
Experiment with different architectures, hyperparameters, and learning rate schedules.
Evaluate the model using quantitative metrics like PSNR and SSIM.
Fine-tune the model based on the characteristics of the specific denoising task.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Function to load and preprocess images
def load_and_preprocess_data(data_dir, image_size):
    images = []
    
    for filename in os.listdir(data_dir):
        img = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_size, image_size))
        img = img / 255.0  # Normalize pixel values to [0, 1]
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        images.append(img)
    
    return np.array(images)

# Load and preprocess noisy and clean images
noisy_images = load_and_preprocess_data("C:\\Users\\abhig\\OneDrive\\Desktop\\CT image Denoising\\noisy", image_size=128)
clean_images = load_and_preprocess_data("C:\\Users\\abhig\\OneDrive\\Desktop\\CT image Denoising\\Clean", image_size=128)

# Build the DnCNN model
def build_dncnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(1, (3, 3), padding='same'))
    return model

model = build_dncnn_model(input_shape=(128, 128, 1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with validation split
history = model.fit(noisy_images, clean_images, batch_size=16, epochs=20, validation_split=0.2)

# Save the trained model
model.save("denoising_model.h5")

# Display training/validation loss curves
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plotting training/validation loss curves
axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Mean Squared Error')
axes[0].legend()
axes[0].set_title('Loss Curves')

new_noisy_image = load_and_preprocess_data("C:\\Users\\abhig\\OneDrive\\Desktop\\CT image Denoising\\Images", image_size=128)
denoised_image = model.predict(new_noisy_image)

# Display the original noisy image
axes[1].imshow(new_noisy_image[0, ..., 0], cmap='gray')
axes[1].set_title('Noisy Image')

# Display the denoised image
axes[2].imshow(denoised_image[0, ..., 0], cmap='gray')
axes[2].set_title('Denoised Image')

plt.tight_layout()
plt.show()
