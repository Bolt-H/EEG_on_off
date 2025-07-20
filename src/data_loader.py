"""
Data loading and preprocessing utilities for Sports Image Classification
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns


class SportsDataLoader:
    """
    Data loader class for Sports Image Classification dataset
    """
    
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize the data loader
        
        Args:
            data_dir (str): Path to the dataset directory
            img_size (tuple): Target image size (height, width)
            batch_size (int): Batch size for data generators
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
    def load_dataset_info(self):
        """
        Load dataset information and create DataFrames
        """
        # Create lists to store file paths and labels
        filepaths = []
        labels = []
        
        # Walk through train directory
        train_dir = os.path.join(self.data_dir, 'train')
        if os.path.exists(train_dir):
            for sport_class in os.listdir(train_dir):
                class_dir = os.path.join(train_dir, sport_class)
                if os.path.isdir(class_dir):
                    for img_file in os.listdir(class_dir):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            filepaths.append(os.path.join(class_dir, img_file))
                            labels.append(sport_class)
        
        # Create DataFrame
        df = pd.DataFrame({'filepath': filepaths, 'label': labels})
        
        # Split into train and validation
        self.train_df, self.val_df = train_test_split(
            df, test_size=0.2, stratify=df['label'], random_state=42
        )
        
        # Load test data if exists
        test_dir = os.path.join(self.data_dir, 'test')
        if os.path.exists(test_dir):
            test_filepaths = []
            test_labels = []
            
            for sport_class in os.listdir(test_dir):
                class_dir = os.path.join(test_dir, sport_class)
                if os.path.isdir(class_dir):
                    for img_file in os.listdir(class_dir):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            test_filepaths.append(os.path.join(class_dir, img_file))
                            test_labels.append(sport_class)
            
            self.test_df = pd.DataFrame({'filepath': test_filepaths, 'label': test_labels})
        
        print(f"Training samples: {len(self.train_df)}")
        print(f"Validation samples: {len(self.val_df)}")
        if self.test_df is not None:
            print(f"Test samples: {len(self.test_df)}")
        print(f"Number of classes: {len(df['label'].unique())}")
        
        return self.train_df, self.val_df, self.test_df
    
    def analyze_class_distribution(self):
        """
        Analyze and visualize class distribution
        """
        if self.train_df is None:
            print("Please load dataset info first using load_dataset_info()")
            return
        
        # Plot class distribution
        plt.figure(figsize=(15, 8))
        
        # Training data distribution
        plt.subplot(1, 2, 1)
        train_counts = self.train_df['label'].value_counts()
        plt.bar(range(len(train_counts)), train_counts.values)
        plt.title('Training Data Class Distribution')
        plt.xlabel('Class Index')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=90)
        
        # Validation data distribution
        plt.subplot(1, 2, 2)
        val_counts = self.val_df['label'].value_counts()
        plt.bar(range(len(val_counts)), val_counts.values)
        plt.title('Validation Data Class Distribution')
        plt.xlabel('Class Index')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=90)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"Training data - Min samples per class: {train_counts.min()}")
        print(f"Training data - Max samples per class: {train_counts.max()}")
        print(f"Training data - Mean samples per class: {train_counts.mean():.2f}")
        
    def create_data_generators(self, augmentation=True):
        """
        Create data generators for training, validation, and testing
        
        Args:
            augmentation (bool): Whether to apply data augmentation to training data
        
        Returns:
            tuple: (train_generator, val_generator, test_generator)
        """
        if self.train_df is None:
            print("Please load dataset info first using load_dataset_info()")
            return None, None, None
        
        # Data augmentation for training
        if augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                shear_range=0.2,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        # No augmentation for validation and test
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_dataframe(
            self.train_df,
            x_col='filepath',
            y_col='label',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_test_datagen.flow_from_dataframe(
            self.val_df,
            x_col='filepath',
            y_col='label',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = None
        if self.test_df is not None:
            test_generator = val_test_datagen.flow_from_dataframe(
                self.test_df,
                x_col='filepath',
                y_col='label',
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False
            )
        
        return train_generator, val_generator, test_generator
    
    def display_sample_images(self, generator, num_images=9):
        """
        Display sample images from a data generator
        
        Args:
            generator: Data generator to sample from
            num_images (int): Number of images to display
        """
        # Get a batch of images
        batch_images, batch_labels = next(generator)
        
        # Get class names
        class_names = list(generator.class_indices.keys())
        
        # Plot images
        plt.figure(figsize=(12, 12))
        for i in range(min(num_images, len(batch_images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(batch_images[i])
            
            # Get predicted class
            predicted_class_idx = np.argmax(batch_labels[i])
            class_name = class_names[predicted_class_idx]
            
            plt.title(f'Class: {class_name}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()


def load_sports_data(data_dir, img_size=(224, 224), batch_size=32, augmentation=True):
    """
    Convenience function to load sports dataset
    
    Args:
        data_dir (str): Path to dataset directory
        img_size (tuple): Target image size
        batch_size (int): Batch size
        augmentation (bool): Whether to apply augmentation
    
    Returns:
        tuple: (data_loader, train_gen, val_gen, test_gen)
    """
    data_loader = SportsDataLoader(data_dir, img_size, batch_size)
    data_loader.load_dataset_info()
    
    train_gen, val_gen, test_gen = data_loader.create_data_generators(augmentation)
    
    return data_loader, train_gen, val_gen, test_gen