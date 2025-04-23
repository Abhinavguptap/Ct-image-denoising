# CT Image Denoising with DnCNN

A lightweight DnCNN model built with TensorFlow to denoise grayscale CT scan images. The model is trained on noisy-clean image pairs and effectively learns to remove noise.

## Features

- DnCNN architecture (3 Conv layers)
- Grayscale preprocessing with OpenCV
- Training & validation loss visualization
- Inference on new noisy images

##  Directory Structure
CT-image-Denoising/ ├── noisy/ # Noisy training images ├── Clean/ # Clean ground truth images ├── Images/ # Noisy images for testing ├── main.py # Main training & inference script └── denoising_model.h5 # Saved model after training

##  How to Use

1. **Install dependencies**
    ```bash
    pip install tensorflow numpy opencv-python matplotlib
    ```

2. **Add your data**  
   Place images in the `noisy/`, `Clean/`, and `Images/` folders.

3. **Run the script**
    ```bash
    python main.py
    ```

4. **View Results**  
   The script will display training loss curves and a side-by-side comparison of noisy vs denoised images.

## Requirements

- TensorFlow  
- NumPy  
- OpenCV  
- Matplotlib

