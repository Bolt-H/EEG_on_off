"""
Model evaluation utilities for Sports Image Classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, top_k_accuracy_score
)
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import load_model
import os


class ModelEvaluator:
    """
    Comprehensive model evaluation class
    """
    
    def __init__(self, model, class_names=None):
        """
        Initialize the evaluator
        
        Args:
            model: Trained Keras model or path to saved model
            class_names (list): List of class names
        """
        if isinstance(model, str):
            self.model = load_model(model)
        else:
            self.model = model
        
        self.class_names = class_names
        self.predictions = None
        self.true_labels = None
        
    def evaluate_on_generator(self, data_generator, steps=None):
        """
        Evaluate model on a data generator
        
        Args:
            data_generator: Keras data generator
            steps (int): Number of steps to evaluate (None for all)
        
        Returns:
            dict: Evaluation metrics
        """
        # Get predictions and true labels
        if steps is None:
            steps = len(data_generator)
        
        # Reset generator
        data_generator.reset()
        
        # Get predictions
        predictions = self.model.predict(data_generator, steps=steps, verbose=1)
        
        # Get true labels
        true_labels = []
        for i in range(steps):
            batch_x, batch_y = next(data_generator)
            true_labels.extend(np.argmax(batch_y, axis=1))
        
        # Convert predictions to class indices
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Store for later use
        self.predictions = predictions
        self.true_labels = np.array(true_labels)
        self.predicted_labels = predicted_labels
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Top-k accuracies
        top_3_acc = top_k_accuracy_score(true_labels, predictions, k=3)
        top_5_acc = top_k_accuracy_score(true_labels, predictions, k=5)
        
        metrics = {
            'accuracy': accuracy,
            'top_3_accuracy': top_3_acc,
            'top_5_accuracy': top_5_acc,
            'total_samples': len(true_labels)
        }
        
        return metrics
    
    def generate_classification_report(self, save_path=None):
        """
        Generate detailed classification report
        
        Args:
            save_path (str): Path to save the report
        
        Returns:
            str: Classification report
        """
        if self.true_labels is None or self.predicted_labels is None:
            raise ValueError("Please run evaluate_on_generator first")
        
        # Generate report
        target_names = self.class_names if self.class_names else None
        report = classification_report(
            self.true_labels, 
            self.predicted_labels,
            target_names=target_names,
            output_dict=False
        )
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def plot_confusion_matrix(self, normalize=True, figsize=(15, 15), save_path=None):
        """
        Plot confusion matrix
        
        Args:
            normalize (bool): Whether to normalize the matrix
            figsize (tuple): Figure size
            save_path (str): Path to save the plot
        """
        if self.true_labels is None or self.predicted_labels is None:
            raise ValueError("Please run evaluate_on_generator first")
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.true_labels, self.predicted_labels)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        # Plot
        plt.figure(figsize=figsize)
        sns.heatmap(cm, 
                   annot=True if len(cm) <= 20 else False,  # Don't annotate if too many classes
                   fmt=fmt,
                   cmap='Blues',
                   xticklabels=self.class_names if self.class_names else range(len(cm)),
                   yticklabels=self.class_names if self.class_names else range(len(cm)))
        
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if len(cm) > 20:
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_per_class_accuracy(self, save_path=None):
        """
        Plot per-class accuracy
        
        Args:
            save_path (str): Path to save the plot
        """
        if self.true_labels is None or self.predicted_labels is None:
            raise ValueError("Please run evaluate_on_generator first")
        
        # Calculate per-class accuracy
        cm = confusion_matrix(self.true_labels, self.predicted_labels)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'class': self.class_names if self.class_names else range(len(per_class_acc)),
            'accuracy': per_class_acc
        })
        df = df.sort_values('accuracy')
        
        # Plot
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(df)), df['accuracy'])
        
        # Color bars based on accuracy
        for i, bar in enumerate(bars):
            if df.iloc[i]['accuracy'] < 0.5:
                bar.set_color('red')
            elif df.iloc[i]['accuracy'] < 0.7:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        plt.xlabel('Classes (sorted by accuracy)')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.xticks(range(len(df)), df['class'], rotation=90)
        plt.grid(True, alpha=0.3)
        
        # Add horizontal line for average accuracy
        avg_acc = per_class_acc.mean()
        plt.axhline(y=avg_acc, color='black', linestyle='--', 
                   label=f'Average: {avg_acc:.3f}')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return df
    
    def analyze_worst_predictions(self, top_k=10):
        """
        Analyze the worst predictions (lowest confidence correct predictions 
        and highest confidence incorrect predictions)
        
        Args:
            top_k (int): Number of worst predictions to analyze
        
        Returns:
            dict: Analysis results
        """
        if self.predictions is None or self.true_labels is None:
            raise ValueError("Please run evaluate_on_generator first")
        
        # Get prediction confidences
        pred_confidences = np.max(self.predictions, axis=1)
        
        # Separate correct and incorrect predictions
        correct_mask = self.predicted_labels == self.true_labels
        
        # Worst correct predictions (lowest confidence)
        correct_indices = np.where(correct_mask)[0]
        correct_confidences = pred_confidences[correct_indices]
        worst_correct_idx = correct_indices[np.argsort(correct_confidences)[:top_k]]
        
        # Worst incorrect predictions (highest confidence)
        incorrect_indices = np.where(~correct_mask)[0]
        incorrect_confidences = pred_confidences[incorrect_indices]
        worst_incorrect_idx = incorrect_indices[np.argsort(incorrect_confidences)[-top_k:]]
        
        analysis = {
            'worst_correct': {
                'indices': worst_correct_idx,
                'confidences': pred_confidences[worst_correct_idx],
                'true_labels': self.true_labels[worst_correct_idx],
                'predicted_labels': self.predicted_labels[worst_correct_idx]
            },
            'worst_incorrect': {
                'indices': worst_incorrect_idx,
                'confidences': pred_confidences[worst_incorrect_idx],
                'true_labels': self.true_labels[worst_incorrect_idx],
                'predicted_labels': self.predicted_labels[worst_incorrect_idx]
            }
        }
        
        return analysis
    
    def plot_training_history(self, history, save_path=None):
        """
        Plot training history
        
        Args:
            history: Keras training history object or dict
            save_path (str): Path to save the plot
        """
        if hasattr(history, 'history'):
            history = history.history
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training & validation accuracy
        if 'accuracy' in history:
            axes[0, 0].plot(history['accuracy'], label='Training Accuracy')
            axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy')
            axes[0, 0].set_title('Model Accuracy')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Plot training & validation loss
        if 'loss' in history:
            axes[0, 1].plot(history['loss'], label='Training Loss')
            axes[0, 1].plot(history['val_loss'], label='Validation Loss')
            axes[0, 1].set_title('Model Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Plot learning rate if available
        if 'lr' in history:
            axes[1, 0].plot(history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True)
        
        # Plot additional metrics if available
        if 'top_k_categorical_accuracy' in history:
            axes[1, 1].plot(history['top_k_categorical_accuracy'], 
                           label='Training Top-K Accuracy')
            axes[1, 1].plot(history['val_top_k_categorical_accuracy'], 
                           label='Validation Top-K Accuracy')
            axes[1, 1].set_title('Top-K Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Top-K Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_evaluation_report(self, data_generator, save_dir='evaluation_results'):
        """
        Generate a comprehensive evaluation report
        
        Args:
            data_generator: Data generator for evaluation
            save_dir (str): Directory to save results
        
        Returns:
            dict: Complete evaluation results
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        print("Evaluating model...")
        
        # Evaluate model
        metrics = self.evaluate_on_generator(data_generator)
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {metrics['top_5_accuracy']:.4f}")
        
        # Generate classification report
        print("\nGenerating classification report...")
        report = self.generate_classification_report(
            os.path.join(save_dir, 'classification_report.txt')
        )
        
        # Plot confusion matrix
        print("Plotting confusion matrix...")
        self.plot_confusion_matrix(
            save_path=os.path.join(save_dir, 'confusion_matrix.png')
        )
        
        # Plot per-class accuracy
        print("Analyzing per-class performance...")
        per_class_df = self.plot_per_class_accuracy(
            save_path=os.path.join(save_dir, 'per_class_accuracy.png')
        )
        
        # Save per-class results
        per_class_df.to_csv(os.path.join(save_dir, 'per_class_accuracy.csv'), index=False)
        
        # Analyze worst predictions
        print("Analyzing worst predictions...")
        worst_analysis = self.analyze_worst_predictions()
        
        # Save results
        results = {
            'metrics': metrics,
            'classification_report': report,
            'per_class_accuracy': per_class_df,
            'worst_predictions': worst_analysis
        }
        
        print(f"Evaluation complete! Results saved to {save_dir}")
        
        return results


def evaluate_model(model_path, data_generator, class_names=None, save_dir='evaluation_results'):
    """
    Convenience function to evaluate a saved model
    
    Args:
        model_path (str): Path to saved model
        data_generator: Data generator for evaluation
        class_names (list): List of class names
        save_dir (str): Directory to save results
    
    Returns:
        dict: Evaluation results
    """
    evaluator = ModelEvaluator(model_path, class_names)
    return evaluator.generate_evaluation_report(data_generator, save_dir)