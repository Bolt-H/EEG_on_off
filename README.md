# Sports Image Classification

A deep learning project for classifying 100 different sports from images using Convolutional Neural Networks (CNN) and Transfer Learning with EfficientNetV2.

## 🏆 Project Overview

This project implements a multi-class image classification system that can identify and classify 100 different types of sports from images. The model uses transfer learning with EfficientNetV2-B0 as the base architecture and achieves high accuracy on sports image recognition tasks.

## 📊 Dataset

- **Total Classes**: 100 different sports categories
- **Training Images**: 15,278 images
- **Validation Images**: 3,820 images  
- **Test Images**: 500 images
- **Image Format**: RGB images, resized to appropriate dimensions for model input

### Sample Sports Categories
- Archery
- Judo
- Tennis
- Water Cycling
- Tug of War
- Sumo Wrestling
- And 94 more...

## 🚀 Features

- **Transfer Learning**: Uses pre-trained EfficientNetV2-B0 for better performance
- **Data Augmentation**: Implements various augmentation techniques to improve model robustness
- **100-Class Classification**: Supports classification of 100 different sports
- **Model Checkpointing**: Saves best performing models during training
- **Visualization**: Includes training/validation loss and accuracy plots

## 📋 Requirements

```
tensorflow>=2.12.0
keras>=2.12.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
Pillow>=8.3.0
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0
```

## 🔧 Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/sports-image-classification.git
cd sports-image-classification
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset and extract it to the project directory

## 🏃‍♂️ Usage

### Option 1: Command Line Training
```bash
# Basic training
python train.py --data_dir data/sports_images_local --epochs 50

# Advanced training
python train.py --model_type efficientnet --epochs 100 --batch_size 64
```

### Option 2: Clean Jupyter Notebook
```bash
jupyter notebook notebooks/sports_classification_clean.ipynb
```

### Option 3: Python API
```python
from src import load_sports_data, create_trainer

# Load data
data_loader, train_gen, val_gen, test_gen = load_sports_data('data/sports_images_local')

# Create trainer and train
trainer = create_trainer('efficientnet', train_generator=train_gen, val_generator=val_gen)
history = trainer.train(epochs=50)
```

### Evaluation
```bash
python evaluate.py --model_path experiments/my_model/best_model.keras --data_dir data/sports_images_local
```

For detailed usage instructions, see [USAGE.md](USAGE.md).

## 📈 Model Architecture

- **Base Model**: EfficientNetV2-B0 (pre-trained on ImageNet)
- **Custom Layers**: 
  - Global Average Pooling 2D
  - Dense layer with 512 units (ReLU activation)
  - Dropout (0.5)
  - Output Dense layer (100 units, softmax activation)

## 🎯 Performance

- **Training Accuracy**: To be determined after running fixed training pipeline
- **Validation Accuracy**: To be determined after running fixed training pipeline  
- **Test Accuracy**: To be determined after evaluation

> **Note**: The original notebook had training execution errors. Use the new modular training pipeline for reliable results.

## 📁 Project Structure

```
sports-image-classification/
├── src/                             # Source code modules
│   ├── data_loader.py              # Data loading utilities
│   ├── model_architecture.py       # Model definitions
│   ├── training.py                 # Training utilities
│   ├── evaluation.py               # Evaluation utilities
│   └── __init__.py
├── notebooks/                       # Jupyter notebooks
│   ├── sports_classification_clean.ipynb  # Clean, organized notebook
│   └── original_notebook.ipynb     # Original notebook (backup)
├── tests/                          # Unit tests
│   ├── test_data_loader.py        # Data loader tests
│   └── __init__.py
├── experiments/                    # Training experiments
├── data/                          # Dataset directory
├── models/                        # Saved models
├── train.py                       # Training script
├── evaluate.py                    # Evaluation script
├── config.py                      # Configuration
├── requirements.txt               # Dependencies
├── USAGE.md                       # Usage guide
└── README.md                      # Project documentation
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- EfficientNet paper and implementation by Google Research
- Sports image dataset contributors
- TensorFlow and Keras communities

## 📞 Contact

- GitHub: [@your-username](https://github.com/your-username)
- Email: your.email@example.com

---
⭐ If you found this project helpful, please give it a star!