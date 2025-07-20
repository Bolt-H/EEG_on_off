"""
Sports Image Classification Package

A comprehensive package for training and evaluating deep learning models
for sports image classification.
"""

from .data_loader import SportsDataLoader, load_sports_data
from .model_architecture import (
    SportsClassificationModel, 
    create_sports_model,
    get_lr_scheduler,
    cosine_decay_with_warmup
)
from .evaluation import ModelEvaluator, evaluate_model
from .training import SportsTrainer, create_trainer

__version__ = "1.0.0"
__author__ = "Sports Classification Team"

__all__ = [
    'SportsDataLoader',
    'load_sports_data',
    'SportsClassificationModel',
    'create_sports_model',
    'get_lr_scheduler',
    'cosine_decay_with_warmup',
    'ModelEvaluator',
    'evaluate_model',
    'SportsTrainer',
    'create_trainer'
]