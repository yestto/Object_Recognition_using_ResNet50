# CIFAR-10 Object Recognition using ResNet50
This project implements a deep learning pipeline to classify images from the CIFAR-10 dataset using a modified ResNet50 architecture. It includes data preprocessing, model customization, training, evaluation, and visualization.

📁 Project Structure
Copy
Edit
DL_Project_4_CIFAR_10_Object_Recognition_using_ResNet50.ipynb
README.md
📦 Features
Loads and preprocesses the CIFAR-10 dataset.

Fine-tunes a pre-trained ResNet50 model on CIFAR-10 (with input shape modifications).

Implements data augmentation for improved generalization.

Includes performance metrics: accuracy, classification report, and confusion matrix.

Visualizes training/validation accuracy and loss.

🧠 Model Architecture
Base model: ResNet50 pre-trained on ImageNet.

Modifications:

Adjusted input shape to 32x32x3 for CIFAR-10.

Replaced top classifier layers with:

Global Average Pooling

Fully connected Dense layers

Dropout for regularization

Final Dense(10) with softmax activation

🧪 Training Details
Optimizer: Adam

Loss: Categorical Crossentropy

Epochs: 20 (modifiable)

Batch Size: 64

EarlyStopping and ModelCheckpoint used to prevent overfitting

📊 Results
Final validation accuracy: ~85–90% (depending on run and tuning)

Includes classification report and confusion matrix

Accuracy/Loss plots for training and validation

🖼️ Sample Predictions
Notebook includes sample predictions with ground truth vs predicted labels.

🚀 Getting Started
Prerequisites
Python ≥ 3.7

TensorFlow ≥ 2.x

NumPy, Matplotlib, scikit-learn

Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/CIFAR10-ResNet50.git
cd CIFAR10-ResNet50
Install dependencies (optional virtual env recommended):

bash
Copy
Edit
pip install -r requirements.txt
Running the Notebook
Launch Jupyter or run directly:

bash
Copy
Edit
jupyter notebook DL_Project_4_CIFAR_10_Object_Recognition_using_ResNet50.ipynb
📚 Dataset
Name: CIFAR-10

Classes: 10 object categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

Size: 60,000 32x32 color images

Automatically downloaded via tf.keras.datasets.cifar10.load_data().

📈 Example Output
Accuracy: ~88%

Loss: ~0.3

Confusion Matrix and Classification Report for detailed metrics
