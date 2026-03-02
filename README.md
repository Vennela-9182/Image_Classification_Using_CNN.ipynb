# Image Classification CNN Model
This repository contains a Jupyter notebook implementing a Convolutional Neural Network (CNN) for binary image classification on a pizza vs. steak dataset. The model uses TensorFlow/Keras to achieve high validation accuracy through iterative improvements like data augmentation and architecture tweaks.

### Dataset
The dataset consists of 1500 training images and 500 validation images across two classes: pizza and steak, sourced from directories like pizza_steak/train and pizza_steak/test. Images are resized to 224x224 pixels and normalized by rescaling pixel values to.

### Model Architecture
The baseline model (Tiny VGG-inspired) is a sequential CNN with:
  - Conv2D layers (10 filters, 3x3 kernel, ReLU activation)
  - MaxPooling2D (2x2 pool size)
  - Flatten and Dense output (sigmoid for binary classification)

Total parameters: ~31K. Enhanced versions add more convolutions or augmentation for better performance.
|Model Variant|Key Changes|Params|Val Accuracy|
|-------------|-----------|------|------------|
|Model 1|`Baseline Tiny VGG`|`31,101`|`30%`|
|Model 5|`	Extra Conv2D + MaxPool`|`8,861`|`87%`|
|Model 8|`Augmented data`|`31,101`|`85%`|

### Training Process
- Data generators: ImageDataGenerator with rescaling=1./255; augmentation includes rotation, zoom, and horizontal flips in later models.
- Optimizer: Adam; Loss: binary_crossentropy; Metrics: accuracy.
- Trained for 5 epochs, batch size 32.
- Example: Early stopping not used, but validation monitoring shows convergence around 85-88% accuracy.

### Performance
Models reach 30-87% validation accuracy after 5 epochs. Loss curves indicate reduced overfitting with augmentation (e.g., Model 8 outperforms baseline). Plots visualize training/validation loss and accuracy trends.

### Usage
- Run Image_Classification_Using_CNN.ipynb in Jupyter/Colab.
- Ensure TensorFlow/Keras installed: pip install tensorflow.
- Download pizza-steak dataset via notebook code.
- Train: model.fit(train_data, epochs=5, validation_data=valid_data).
- Predict on new images using trained model

### Improvements & Next Steps
- Addresses overfitting via augmentation and deeper layers.
- Potential: Add dropout, early stopping, or transfer learning (e.g., ResNet).
