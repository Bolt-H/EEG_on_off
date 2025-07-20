#!/usr/bin/env python3
"""
Training script for Sports Image Classification

Usage:
    python train.py --data_dir /path/to/data --epochs 50 --batch_size 32
"""

import argparse
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.data_loader import load_sports_data
from src.training import create_trainer
from config import setup_environment, get_data_dir, get_models_dir


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Train Sports Image Classification Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224],
                       help='Image size (height width)')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='efficientnet',
                       choices=['efficientnet', 'resnet', 'custom'],
                       help='Type of model to use')
    parser.add_argument('--num_classes', type=int, default=100,
                       help='Number of classes')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--trainable_base', action='store_true',
                       help='Make base model trainable from start')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--monitor', type=str, default='val_accuracy',
                       help='Metric to monitor for early stopping')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                       choices=['cosine', 'step', 'exponential'],
                       help='Learning rate schedule')
    
    # Experiment arguments
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Other arguments
    parser.add_argument('--augmentation', action='store_true', default=True,
                       help='Use data augmentation')
    parser.add_argument('--no_augmentation', dest='augmentation', action='store_false',
                       help='Disable data augmentation')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level')
    
    return parser.parse_args()


def main():
    """
    Main training function
    """
    args = parse_args()
    
    # Setup environment
    setup_environment()
    
    # Determine data directory
    data_dir = args.data_dir if args.data_dir else get_data_dir()
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist!")
        print("Please provide a valid data directory using --data_dir")
        return
    
    print("=" * 60)
    print("SPORTS IMAGE CLASSIFICATION TRAINING")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Augmentation: {args.augmentation}")
    print("=" * 60)
    
    try:
        # Load data
        print("Loading dataset...")
        data_loader, train_gen, val_gen, test_gen = load_sports_data(
            data_dir=data_dir,
            img_size=tuple(args.img_size),
            batch_size=args.batch_size,
            augmentation=args.augmentation
        )
        
        # Create experiment name if not provided
        if args.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.experiment_name = f"sports_{args.model_type}_{timestamp}"
        
        # Create trainer
        print("Creating trainer...")
        trainer = create_trainer(
            model_type=args.model_type,
            num_classes=args.num_classes,
            img_size=(*args.img_size, 3),
            train_generator=train_gen,
            val_generator=val_gen,
            test_generator=test_gen,
            experiment_name=args.experiment_name,
            trainable_base=args.trainable_base,
            dropout_rate=args.dropout_rate
        )
        
        # Compile model with specified learning rate
        trainer.model_builder.compile_model(
            learning_rate=args.learning_rate
        )
        
        # Resume from checkpoint if specified
        if args.resume_from:
            print(f"Resuming from checkpoint: {args.resume_from}")
            # Implementation for resuming would go here
        
        # Train the model
        print("Starting training...")
        history = trainer.train(
            epochs=args.epochs,
            patience=args.patience,
            monitor=args.monitor,
            learning_rate_schedule=args.lr_schedule
        )
        
        # Plot training history
        print("Plotting training history...")
        trainer.plot_training_history()
        
        # Evaluate model
        print("Evaluating model...")
        results = trainer.evaluate_model(
            use_test_data=True,
            save_results=True
        )
        
        # Print final results
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Experiment: {args.experiment_name}")
        print(f"Best model saved: {trainer.best_model_path}")
        print(f"Final accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"Top-3 accuracy: {results['metrics']['top_3_accuracy']:.4f}")
        print(f"Top-5 accuracy: {results['metrics']['top_5_accuracy']:.4f}")
        print(f"Results directory: {trainer.experiment_dir}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        return
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()