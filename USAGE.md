# Usage Guide - Sports Image Classification

This guide shows you how to use the Sports Image Classification project for training and evaluating models.

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd sports-image-classification

# Install dependencies
pip install -r requirements.txt

# Setup project directories
python -c "from config import setup_environment; setup_environment()"
```

### 2. Prepare Your Data

Organize your data in the following structure:
```
data/
├── sports_images_local/
│   ├── train/
│   │   ├── sport1/
│   │   │   ├── img1.jpg
│   │   │   └── img2.jpg
│   │   └── sport2/
│   │       ├── img1.jpg
│   │       └── img2.jpg
│   └── test/
│       ├── sport1/
│       │   └── test_img1.jpg
│       └── sport2/
│           └── test_img1.jpg
```

### 3. Train a Model

#### Option A: Using the Training Script
```bash
# Basic training
python train.py --data_dir data/sports_images_local --epochs 50

# Advanced training with custom parameters
python train.py \
    --data_dir data/sports_images_local \
    --model_type efficientnet \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --experiment_name my_sports_model
```

#### Option B: Using the Clean Notebook
Open and run `notebooks/sports_classification_clean.ipynb` in Jupyter.

#### Option C: Using Python API
```python
from src import load_sports_data, create_trainer

# Load data
data_loader, train_gen, val_gen, test_gen = load_sports_data(
    data_dir='data/sports_images_local',
    img_size=(224, 224),
    batch_size=32
)

# Create and train model
trainer = create_trainer(
    model_type='efficientnet',
    train_generator=train_gen,
    val_generator=val_gen,
    test_generator=test_gen,
    experiment_name='my_experiment'
)

# Train
history = trainer.train(epochs=50)

# Evaluate
results = trainer.evaluate_model()
```

### 4. Evaluate a Trained Model

```bash
python evaluate.py \
    --model_path experiments/my_experiment/best_model.keras \
    --data_dir data/sports_images_local \
    --save_dir evaluation_results
```

## Advanced Usage

### Custom Model Architecture

```python
from src.model_architecture import SportsClassificationModel

# Create custom model
model_builder = SportsClassificationModel(num_classes=100)

# Build different architectures
efficientnet_model = model_builder.build_efficientnet_model(trainable_base=False)
resnet_model = model_builder.build_resnet_model(trainable_base=True)
custom_model = model_builder.build_custom_cnn()
```

### Fine-tuning

```python
# After initial training
ft_history = trainer.fine_tune(
    unfreeze_layers=20,
    new_learning_rate=1e-5,
    epochs=10
)
```

### Data Analysis

```python
from src.data_loader import SportsDataLoader

loader = SportsDataLoader('data/sports_images_local')
loader.load_dataset_info()
loader.analyze_class_distribution()
```

### Comprehensive Evaluation

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator('path/to/model.keras', class_names)
results = evaluator.generate_evaluation_report(test_generator)

# Plot confusion matrix
evaluator.plot_confusion_matrix()

# Analyze per-class performance
evaluator.plot_per_class_accuracy()
```

## Command Line Options

### Training Script Options

```bash
python train.py --help
```

Key options:
- `--data_dir`: Path to dataset
- `--model_type`: Model architecture (efficientnet, resnet, custom)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--learning_rate`: Initial learning rate
- `--experiment_name`: Name for experiment tracking
- `--patience`: Early stopping patience
- `--augmentation/--no_augmentation`: Enable/disable data augmentation

### Evaluation Script Options

```bash
python evaluate.py --help
```

Key options:
- `--model_path`: Path to trained model file
- `--data_dir`: Path to dataset
- `--use_test_data/--use_val_data`: Choose evaluation dataset
- `--save_dir`: Directory to save results

## Project Structure

```
sports-image-classification/
├── src/                          # Source code modules
│   ├── data_loader.py           # Data loading utilities
│   ├── model_architecture.py    # Model definitions
│   ├── training.py              # Training utilities
│   ├── evaluation.py            # Evaluation utilities
│   └── __init__.py
├── notebooks/                    # Jupyter notebooks
│   ├── sports_classification_clean.ipynb
│   └── original_notebook.ipynb
├── tests/                        # Unit tests
├── experiments/                  # Training experiments
├── data/                        # Dataset directory
├── models/                      # Saved models
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── config.py                    # Configuration
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation
```

## Configuration

Edit `config.py` to customize:
- Default paths
- Model parameters
- Training settings
- Environment-specific configurations

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**
   - Reduce batch size
   - Use mixed precision training
   - Enable gradient checkpointing

2. **Data Loading Errors**
   - Check data directory structure
   - Verify image file extensions
   - Ensure proper permissions

3. **Training Convergence Issues**
   - Adjust learning rate
   - Try different optimizers
   - Check data quality and balance

### Getting Help

1. Check the documentation in `README.md`
2. Look at example notebooks
3. Run unit tests: `python -m pytest tests/`
4. Check experiment logs in `experiments/` directory

## Best Practices

1. **Data Preparation**
   - Ensure balanced classes
   - Use proper train/validation/test splits
   - Apply appropriate data augmentation

2. **Training**
   - Start with pre-trained models
   - Use early stopping
   - Monitor both training and validation metrics
   - Save experiment configurations

3. **Evaluation**
   - Use multiple metrics (accuracy, top-k, per-class)
   - Analyze confusion matrices
   - Test on held-out data
   - Document results and findings

4. **Experimentation**
   - Use meaningful experiment names
   - Track hyperparameters
   - Compare multiple model architectures
   - Version control your code and data