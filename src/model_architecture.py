"""
Model architecture definitions for Sports Image Classification
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Activation, 
    GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten
)
from tensorflow.keras.applications import (
    EfficientNetV2B0, ResNet50, VGG16, InceptionV3, MobileNetV2
)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
    LearningRateScheduler
)
import math


class SportsClassificationModel:
    """
    Sports Image Classification Model Builder
    """
    
    def __init__(self, num_classes=100, img_size=(224, 224, 3)):
        """
        Initialize the model builder
        
        Args:
            num_classes (int): Number of sports classes
            img_size (tuple): Input image shape (height, width, channels)
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        
    def build_efficientnet_model(self, trainable_base=False, dropout_rate=0.5):
        """
        Build EfficientNetV2B0-based model
        
        Args:
            trainable_base (bool): Whether to make base model trainable
            dropout_rate (float): Dropout rate for regularization
        
        Returns:
            keras.Model: Compiled model
        """
        # Load pre-trained EfficientNetV2B0
        base_model = EfficientNetV2B0(
            input_shape=self.img_size,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze or unfreeze base model
        base_model.trainable = trainable_base
        
        # Build the model
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dense(512, activation='relu'),
            Dropout(dropout_rate),
            BatchNormalization(),
            Dense(256, activation='relu'),
            Dropout(dropout_rate/2),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_resnet_model(self, trainable_base=False, dropout_rate=0.5):
        """
        Build ResNet50-based model
        
        Args:
            trainable_base (bool): Whether to make base model trainable
            dropout_rate (float): Dropout rate for regularization
        
        Returns:
            keras.Model: Compiled model
        """
        base_model = ResNet50(
            input_shape=self.img_size,
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = trainable_base
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dense(512, activation='relu'),
            Dropout(dropout_rate),
            Dense(256, activation='relu'),
            Dropout(dropout_rate/2),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_custom_cnn(self):
        """
        Build a custom CNN from scratch
        
        Returns:
            keras.Model: Compiled model
        """
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), input_shape=self.img_size),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(2, 2),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(2, 2),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(2, 2),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(2, 2),
            
            # Classifier
            Flatten(),
            Dense(512),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.5),
            Dense(256),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, optimizer='adam', learning_rate=1e-4, metrics=['accuracy']):
        """
        Compile the model
        
        Args:
            optimizer (str or optimizer): Optimizer to use
            learning_rate (float): Learning rate
            metrics (list): List of metrics to track
        """
        if self.model is None:
            raise ValueError("Model not built yet. Please build a model first.")
        
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=metrics
        )
        
        return self.model
    
    def get_callbacks(self, model_save_path='models/best_sports_model.keras', 
                     patience=10, monitor='val_accuracy'):
        """
        Get training callbacks
        
        Args:
            model_save_path (str): Path to save the best model
            patience (int): Early stopping patience
            monitor (str): Metric to monitor
        
        Returns:
            list: List of callbacks
        """
        callbacks = [
            ModelCheckpoint(
                model_save_path,
                monitor=monitor,
                save_best_only=True,
                mode='max' if 'accuracy' in monitor else 'min',
                verbose=1
            ),
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor=monitor,
                factor=0.2,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def get_model_summary(self):
        """
        Print model summary
        """
        if self.model is None:
            print("Model not built yet.")
            return
        
        self.model.summary()
        
        # Calculate total parameters
        total_params = self.model.count_params()
        print(f"\nTotal parameters: {total_params:,}")
        
        # Calculate model size in MB (approximate)
        model_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per parameter
        print(f"Approximate model size: {model_size_mb:.2f} MB")


def create_sports_model(model_type='efficientnet', num_classes=100, 
                       img_size=(224, 224, 3), **kwargs):
    """
    Convenience function to create and compile a sports classification model
    
    Args:
        model_type (str): Type of model ('efficientnet', 'resnet', 'custom')
        num_classes (int): Number of classes
        img_size (tuple): Input image shape
        **kwargs: Additional arguments for model building
    
    Returns:
        tuple: (model_builder, compiled_model)
    """
    builder = SportsClassificationModel(num_classes, img_size)
    
    if model_type.lower() == 'efficientnet':
        model = builder.build_efficientnet_model(**kwargs)
    elif model_type.lower() == 'resnet':
        model = builder.build_resnet_model(**kwargs)
    elif model_type.lower() == 'custom':
        model = builder.build_custom_cnn()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Compile the model
    builder.compile_model()
    
    return builder, model


def cosine_decay_with_warmup(epoch, total_epochs=100, warmup_epochs=10, 
                            initial_lr=1e-4, min_lr=1e-6):
    """
    Cosine decay learning rate schedule with warmup
    
    Args:
        epoch (int): Current epoch
        total_epochs (int): Total number of epochs
        warmup_epochs (int): Number of warmup epochs
        initial_lr (float): Initial learning rate
        min_lr (float): Minimum learning rate
    
    Returns:
        float: Learning rate for the current epoch
    """
    if epoch < warmup_epochs:
        # Warmup phase
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay phase
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (initial_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def get_lr_scheduler(schedule_type='cosine', **kwargs):
    """
    Get learning rate scheduler callback
    
    Args:
        schedule_type (str): Type of schedule ('cosine', 'step', 'exponential')
        **kwargs: Additional arguments for the scheduler
    
    Returns:
        LearningRateScheduler: Callback for learning rate scheduling
    """
    if schedule_type == 'cosine':
        def lr_schedule(epoch):
            return cosine_decay_with_warmup(epoch, **kwargs)
    elif schedule_type == 'step':
        def lr_schedule(epoch):
            initial_lr = kwargs.get('initial_lr', 1e-4)
            drop_rate = kwargs.get('drop_rate', 0.5)
            epochs_drop = kwargs.get('epochs_drop', 20)
            return initial_lr * math.pow(drop_rate, math.floor(epoch / epochs_drop))
    elif schedule_type == 'exponential':
        def lr_schedule(epoch):
            initial_lr = kwargs.get('initial_lr', 1e-4)
            decay_rate = kwargs.get('decay_rate', 0.95)
            return initial_lr * math.pow(decay_rate, epoch)
    else:
        raise ValueError(f"Unsupported schedule type: {schedule_type}")
    
    return LearningRateScheduler(lr_schedule, verbose=1)