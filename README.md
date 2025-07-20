My Machine Learning Model
# 1. Sports Image Classification 

A deep learning project for classifying 100 different sports from images using CNN

## 🏆 Project Overview

This Project implements a multi-class image classification system that can identify and classify 100 different types of sports from images. The model uses transfer learning with EfficientNetV2-80 as the base architecture and I am high accuracy on sports image recognition tasks.

## 📊 Dataset
- **Total Class**: 100 different sports categories
- **Training Images**: 15280 images
- **Validation Images**: 3820 images
- **Test Images**: 500 images
- **Images Format**: RGB images, resized to appropriate dimensions for model input

### Sample
- Archery
- Judo
- Tennis
- And ...

## 🚀 Features
- **Transfer Learning**: Uses pre-trained EfficientNetV2-B0 for better performance
- **Data Augmentation**: Implements various augmentation techniques to improve model robustness
- **100-Class Classification**: Supports classification of 100 different sports 
- **Model Checkpointing**: Saves best performing models during training 
- **Visualization**: Includes training/valdation loss and accuracy plots

## 🧾 Requirements
 ```
tensorflow>=tensorflow>=2.12.0
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
 
 ### Training the Model
 ```python
 # Run the Jupyter notebook
 jupyter notebook Sports_Image_classification.ipynb
 ```
 
 ### Making Predictions
 ```python
 # Load the trained model and make predictions on new images
 # (Implementation details in the notebook)
 ```
 

 ## 📈 Model Architecture
 
 - **Base Model**: EfficientNetV2-B0 (pre-trained on ImageNet)
 - **Custom Layers**: 
   - Global Average Pooling 2D
   - Dense layer with 512 units (ReLU activation)
   - Dropout (0.5)
   - Output Dense layer (100 units, softmax activation)
 
 ## 🎯 Performance
 
 - **Training Accuracy**: [To be updated after training completion]
 - **Validation Accuracy**: [To be updated after training completion]
 - **Test Accuracy**: [To be updated after evaluation]
 
 ## 📁 Project Structure
 
 ```
 sports-image-classification/
 ├── Sports_Image_classification.ipynb  # Main notebook
 ├── README.md                         # Project documentation
 ├── requirements.txt                  # Dependencies
 ├── models/                          # Saved models (to be created)
 ├── data/                           # Dataset directory (to be created)
 └── src/                           # Source code modules (to be created)
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
 
 - GitHub: [@Bolt-H](https://github.com/Bolt-H)
 - Email: chayanonputpanthasak@gmail.com
 
 ---
 ⭐ If you found this project helpful, please give it a star!
