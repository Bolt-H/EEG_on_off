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

    Arguments:
      model: Trained Keras model or path to saved model 
      class_names (list): List of class names
    """
    if isinstance(model, str):
      self.model = load_model(model)
    else:
      self.madel = model
    self.class_names = class_names
    self.predictions = None
    self.true_labels = None
  def evaluate_on_generator(self, data_generator, steps=None):
    """
    Evaluate model on a data generator

    Arguments:
      data_generator: Keras data generator
      steps (int): Number of steps to evaluate (None for all)
    Returns:
      dict: Evaluation metrics
    """
    # Get predictions and true labels
    if steps is None::
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
    self.predictions = prediction
    self.true_labels = np.array(true_labels)
    self.predicte_labels = predicted_labels
    # Calculate metrics 
    accuracy = accuracy_score(true_label, predicted_labels)
    # Top-k accuracies
    top_3_acc = top_k_accuracy_score(true_labels, prediction, k=3)
    top_5_acc = top_k_accuracy_score(true_labels, prediction, k=5)
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

    Arguments:
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
    
