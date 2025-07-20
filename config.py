"""
Configuration file for Sports Image Classification project
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, 'notebooks')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

# Data configuration
DEFAULT_DATA_DIR = os.path.join(DATA_DIR, 'sports_images_local')
TRAIN_DIR = os.path.join(DEFAULT_DATA_DIR, 'train')
TEST_DIR = os.path.join(DEFAULT_DATA_DIR, 'test')
VAL_DIR = os.path.join(DEFAULT_DATA_DIR, 'val')

# Model configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 100
CHANNELS = 3

# Training configuration
DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 10
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_DROPOUT_RATE = 0.5

# Supported model types
SUPPORTED_MODELS = ['efficientnet', 'resnet', 'custom']

# File extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Environment settings
def setup_environment():
    """
    Set up the environment for the project
    """
    # Create necessary directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
    
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Experiments directory: {EXPERIMENTS_DIR}")

# Google Colab specific paths (for compatibility)
COLAB_PATHS = {
    'drive_mount': '/content/drive',
    'data_dir': '/content/sports_images_local',
    'models_dir': '/content/models'
}

def is_colab():
    """
    Check if running in Google Colab
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False

def get_data_dir():
    """
    Get the appropriate data directory based on environment
    """
    if is_colab():
        return COLAB_PATHS['data_dir']
    else:
        return DEFAULT_DATA_DIR

def get_models_dir():
    """
    Get the appropriate models directory based on environment
    """
    if is_colab():
        return COLAB_PATHS['models_dir']
    else:
        return MODELS_DIR