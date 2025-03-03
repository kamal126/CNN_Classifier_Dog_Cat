# 🐱🐶 CNN Classifier - Cats vs. Dogs  

## 📂 Dataset  
**Source:** Built-in TensorFlow dataset (`tf.keras.datasets`)  

### 🔹 Data Loading  
```python
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
```
🔹 Preprocessing
Resized images to (150, 150)
Normalized pixel values to the range [0,1]

🏗 Model Architecture
Type: Convolutional Neural Network (CNN)

🔹 Layer Composition
1️⃣ Conv2D → ReLU → MaxPooling
2️⃣ Conv2D → ReLU → MaxPooling
3️⃣ Flatten → Dense → Dropout
4️⃣ Output Layer: Sigmoid activation for binary classification

## 📦 Required Libraries
```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
```
## 🎯 Model Training
Optimizer: Adam
Loss Function: Sparse Categorical Cross-Entropy
Epochs: 5
Validation: Utilized a portion of the dataset for validation

### 💾 Model Saving
The trained model is stored in HDF5 format (.h5), enabling seamless reusability for inference without the need for retraining.

### 🚀 Model Loading & Inference
The saved .h5 model can be reloaded effortlessly and utilized for making predictions on unseen images without requiring recompilation.

✅ This CNN-based classifier efficiently distinguishes between images of cats and dogs, leveraging deep learning techniques for accurate classification.


