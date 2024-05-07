**README: Custom CNN Model for MNIST Classification**

---

### Overview:
This repository contains the code for training a custom Convolutional Neural Network (CNN) model on the MNIST dataset for handwritten digit classification. The CNN architecture is designed from scratch using TensorFlow and Keras.

### Project Structure:
- **src**: This directory contains the source code for the custom CNN model.
  - `cnn_model.py`: Python script containing the implementation of the CNN model.
- **README.md**: Provides an overview of the project, its structure, and instructions for running the code.
- **requirements.txt**: Lists Python dependencies required for the project.

### Dependencies:
Ensure you have Python installed on your system. You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Usage:
1. Clone this repository to your local machine:

```bash
git clone https://github.com/muhammadumair862/Custom-CNN-Model-for-MNIST-Classification.git
```

2. Navigate to the project directory:

```bash
cd Custom-CNN-Model-for-MNIST-Classification
```

3. Open and run the `cnn_model.py` script to train the custom CNN model on the MNIST dataset.

### Model Architecture:
- The custom CNN model architecture consists of convolutional layers followed by max-pooling layers, dropout layers for regularization, and dense layers for classification.
- Activation function: ReLU for hidden layers, softmax for the output layer.
- Loss function: Sparse categorical cross-entropy.
- Optimizer: Adam optimizer.

### Results:
- The custom CNN model is trained to classify handwritten digits with high accuracy on the MNIST dataset.
- Performance metrics such as accuracy, loss, and validation scores are evaluated to assess the model's effectiveness.