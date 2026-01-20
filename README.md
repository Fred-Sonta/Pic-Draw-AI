# Pic-Draw-AI

**Pic-Draw-AI** is a deep learning project designed to recognize hand-drawn sketches. It provides an interface to draw shapes, collect training data, and train a neural network model to classify the drawings in real-time.

## 📂 Project Structure

* `main_app.py`: The main application where users can draw and see predictions.
* `train_model.py`: Script to train the Neural Network using the dataset.
* `collect_data.py`: Utility to draw and save images to build your own dataset.
* `dataset/`: Contains the training images organized by category (Square, Circle, Star, House, etc.).
* `my_demo_model.h5`: The pre-trained model file.
* `drawing_utils.py`: Helper functions for handling canvas drawing and image processing.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed along with the necessary libraries (typically TensorFlow/Keras, NumPy, OpenCV).

```bash
pip install tensorflow numpy opencv-python matplotlib
1. Running the Application
To start the drawing interface and test the current model:
