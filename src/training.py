"""
Training utilities for Sports Image Classification
"""

import os
import json
import pickle
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt

from .model_architecture import SportsClassificationModel, get_lr_scheduler
from .evaluation import ModelEvaluator


class SportsTrainer:
    """
    Training manager for Sports Image Classification
    """
    
    def __init__(self, model_builder, train_generator, val_generator, 
                 test_generator=None, experiment_name=None):
        """
        Initialize the trainer
        
        Args:
            model_builder: SportsClassificationModel instance
            train_generator: Training data generator
            val_generator: Validation data generator
            test_generator: Test data generator (optional)
            experiment_name (str): Name for this training experiment
        """
        self.model_builder = model_builder
        self.model = model_builder.model
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator
        
        # Create experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"sports_classification_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join('experiments', experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Training history
        self.history = None
        self.best_model_path = None
        
    def save_experiment_config(self, config):
        """
        Save experiment configuration
        
        Args:
            config (dict): Configuration dictionary
        """
        config_path = os.path.join(self.experiment_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def train(self, epochs=50, initial_epochs=0, callbacks=None, 
              save_best_only=True, monitor='val_accuracy', patience=10,
              learning_rate_schedule=None, **fit_kwargs):
        """
        Train the model
        
        Args:
            epochs (int): Number of epochs to train
            initial_epochs (int): Initial epoch number (for resuming training)
            callbacks (list): Additional callbacks
            save_best_only (bool): Whether to save only the best model
            monitor (str): Metric to monitor for saving best model
            patience (int): Early stopping patience
            learning_rate_schedule (str): Type of learning rate schedule
            **fit_kwargs: Additional arguments for model.fit()
        
        Returns:
            History: Training history
        """
        print(f"Starting training experiment: {self.experiment_name}")
        print(f"Experiment directory: {self.experiment_dir}")
        
        # Prepare callbacks
        callback_list = []
        
        # Model checkpoint
        self.best_model_path = os.path.join(self.experiment_dir, 'best_model.keras')
        checkpoint = ModelCheckpoint(
            self.best_model_path,
            monitor=monitor,
            save_best_only=save_best_only,
            mode='max' if 'accuracy' in monitor else 'min',
            verbose=1
        )
        callback_list.append(checkpoint)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor=monitor,
            factor=0.2,
            patience=patience//2,
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # Learning rate scheduler
        if learning_rate_schedule:
            lr_scheduler = get_lr_scheduler(
                learning_rate_schedule,
                total_epochs=epochs,
                initial_lr=self.model.optimizer.learning_rate.numpy()
            )
            callback_list.append(lr_scheduler)
        
        # Add custom callbacks
        if callbacks:
            callback_list.extend(callbacks)
        
        # Calculate steps
        steps_per_epoch = len(self.train_generator)
        validation_steps = len(self.val_generator)
        
        # Save training configuration
        config = {
            'model_type': 'efficientnet',  # This should be parameterized
            'epochs': epochs,
            'initial_epochs': initial_epochs,
            'steps_per_epoch': steps_per_epoch,
            'validation_steps': validation_steps,
            'batch_size': self.train_generator.batch_size,
            'image_size': self.train_generator.target_size,
            'optimizer': self.model.optimizer.get_config(),
            'monitor': monitor,
            'patience': patience,
            'learning_rate_schedule': learning_rate_schedule,
            'timestamp': datetime.now().isoformat()
        }
        self.save_experiment_config(config)
        
        # Train the model
        print(f"Training for {epochs} epochs...")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            initial_epoch=initial_epochs,
            validation_data=self.val_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callback_list,
            verbose=1,
            **fit_kwargs
        )
        
        # Save training history
        self.save_training_history()
        
        print(f"Training completed! Best model saved to: {self.best_model_path}")
        
        return self.history
    
    def save_training_history(self):
        """
        Save training history
        """
        if self.history is None:
            return
        
        # Save as pickle
        history_path = os.path.join(self.experiment_dir, 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.history.history, f)
        
        # Save as JSON (for easier reading)
        history_json_path = os.path.join(self.experiment_dir, 'training_history.json')
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        with open(history_json_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
    
    def plot_training_history(self, save_plot=True):
        """
        Plot and optionally save training history
        
        Args:
            save_plot (bool): Whether to save the plot
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        evaluator = ModelEvaluator(self.model)
        plot_path = os.path.join(self.experiment_dir, 'training_history.png') if save_plot else None
        evaluator.plot_training_history(self.history, plot_path)
    
    def evaluate_model(self, use_test_data=True, save_results=True):
        """
        Evaluate the trained model
        
        Args:
            use_test_data (bool): Whether to use test data for evaluation
            save_results (bool): Whether to save evaluation results
        
        Returns:
            dict: Evaluation results
        """
        if self.best_model_path is None or not os.path.exists(self.best_model_path):
            print("No trained model found. Train the model first.")
            return None
        
        # Load the best model
        best_model = tf.keras.models.load_model(self.best_model_path)
        
        # Choose evaluation data
        eval_generator = self.test_generator if (use_test_data and self.test_generator) else self.val_generator
        eval_name = "test" if (use_test_data and self.test_generator) else "validation"
        
        print(f"Evaluating model on {eval_name} data...")
        
        # Get class names from generator
        class_names = list(eval_generator.class_indices.keys())
        
        # Create evaluator
        evaluator = ModelEvaluator(best_model, class_names)
        
        # Evaluate
        if save_results:
            results_dir = os.path.join(self.experiment_dir, f'{eval_name}_evaluation')
            results = evaluator.generate_evaluation_report(eval_generator, results_dir)
        else:
            results = evaluator.evaluate_on_generator(eval_generator)
        
        return results
    
    def fine_tune(self, unfreeze_layers=None, new_learning_rate=1e-5, 
                  epochs=10, **train_kwargs):
        """
        Fine-tune the model by unfreezing some layers
        
        Args:
            unfreeze_layers (int): Number of top layers to unfreeze (None for all)
            new_learning_rate (float): New learning rate for fine-tuning
            epochs (int): Number of epochs for fine-tuning
            **train_kwargs: Additional training arguments
        
        Returns:
            History: Fine-tuning history
        """
        if self.best_model_path is None:
            print("No trained model found. Train the model first.")
            return None
        
        print("Starting fine-tuning...")
        
        # Load the best model
        self.model = tf.keras.models.load_model(self.best_model_path)
        
        # Unfreeze layers
        base_model = self.model.layers[0]  # Assuming first layer is the base model
        
        if unfreeze_layers is None:
            # Unfreeze all layers
            base_model.trainable = True
        else:
            # Unfreeze top N layers
            base_model.trainable = True
            for layer in base_model.layers[:-unfreeze_layers]:
                layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=new_learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Update experiment name for fine-tuning
        ft_experiment_name = f"{self.experiment_name}_finetune"
        self.experiment_name = ft_experiment_name
        self.experiment_dir = os.path.join('experiments', ft_experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Fine-tune
        ft_history = self.train(
            epochs=epochs,
            initial_epochs=0,
            learning_rate_schedule=None,  # Usually no LR schedule for fine-tuning
            **train_kwargs
        )
        
        return ft_history
    
    def resume_training(self, additional_epochs=10):
        """
        Resume training from the best checkpoint
        
        Args:
            additional_epochs (int): Number of additional epochs to train
        
        Returns:
            History: Additional training history
        """
        if self.best_model_path is None or not os.path.exists(self.best_model_path):
            print("No checkpoint found to resume from.")
            return None
        
        # Load the best model
        self.model = tf.keras.models.load_model(self.best_model_path)
        
        # Get the initial epoch from history
        initial_epochs = len(self.history.history['loss']) if self.history else 0
        
        print(f"Resuming training from epoch {initial_epochs}...")
        
        # Continue training
        additional_history = self.train(
            epochs=initial_epochs + additional_epochs,
            initial_epochs=initial_epochs
        )
        
        return additional_history


def create_trainer(model_type='efficientnet', num_classes=100, img_size=(224, 224, 3),
                  train_generator=None, val_generator=None, test_generator=None,
                  experiment_name=None, **model_kwargs):
    """
    Convenience function to create a trainer
    
    Args:
        model_type (str): Type of model to create
        num_classes (int): Number of classes
        img_size (tuple): Input image size
        train_generator: Training data generator
        val_generator: Validation data generator
        test_generator: Test data generator
        experiment_name (str): Experiment name
        **model_kwargs: Additional model arguments
    
    Returns:
        SportsTrainer: Configured trainer instance
    """
    # Create model
    model_builder = SportsClassificationModel(num_classes, img_size)
    
    if model_type.lower() == 'efficientnet':
        model = model_builder.build_efficientnet_model(**model_kwargs)
    elif model_type.lower() == 'resnet':
        model = model_builder.build_resnet_model(**model_kwargs)
    elif model_type.lower() == 'custom':
        model = model_builder.build_custom_cnn()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Compile model
    model_builder.compile_model()
    
    # Create trainer
    trainer = SportsTrainer(
        model_builder, train_generator, val_generator, 
        test_generator, experiment_name
    )
    
    return trainer