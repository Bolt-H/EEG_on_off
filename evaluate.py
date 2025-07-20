#!/usr/bin/env python3
"""
Evaluation script for Sports Image Classification

Usage:
    python evaluate.py --model_path /path/to/model.keras --data_dir /path/to/data
"""

import argparse
import sys
import os

# Add src to path
sys.path.append('src')

from src.data_loader import load_sports_data
from src.evaluation import evaluate_model
from config import get_data_dir


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate Sports Image Classification Model')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file (.keras)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to dataset directory')
    
    # Data arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224],
                       help='Image size (height width)')
    parser.add_argument('--use_test_data', action='store_true', default=True,
                       help='Use test data for evaluation')
    parser.add_argument('--use_val_data', dest='use_test_data', action='store_false',
                       help='Use validation data for evaluation')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--save_plots', action='store_true', default=True,
                       help='Save evaluation plots')
    parser.add_argument('--no_plots', dest='save_plots', action='store_false',
                       help='Do not save evaluation plots')
    
    return parser.parse_args()


def main():
    """
    Main evaluation function
    """
    args = parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist!")
        return
    
    # Determine data directory
    data_dir = args.data_dir if args.data_dir else get_data_dir()
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist!")
        print("Please provide a valid data directory using --data_dir")
        return
    
    print("=" * 60)
    print("SPORTS IMAGE CLASSIFICATION EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data directory: {data_dir}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Using {'test' if args.use_test_data else 'validation'} data")
    print("=" * 60)
    
    try:
        # Load data
        print("Loading dataset...")
        data_loader, train_gen, val_gen, test_gen = load_sports_data(
            data_dir=data_dir,
            img_size=tuple(args.img_size),
            batch_size=args.batch_size,
            augmentation=False  # No augmentation for evaluation
        )
        
        # Choose evaluation data
        eval_generator = test_gen if (args.use_test_data and test_gen) else val_gen
        eval_name = "test" if (args.use_test_data and test_gen) else "validation"
        
        print(f"Evaluating on {eval_name} data...")
        print(f"Number of samples: {eval_generator.samples}")
        print(f"Number of classes: {eval_generator.num_classes}")
        
        # Get class names
        class_names = list(eval_generator.class_indices.keys())
        
        # Evaluate model
        print("Starting evaluation...")
        results = evaluate_model(
            model_path=args.model_path,
            data_generator=eval_generator,
            class_names=class_names,
            save_dir=args.save_dir
        )
        
        # Print results
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED!")
        print("=" * 60)
        print(f"Dataset: {eval_name}")
        print(f"Total samples: {results['metrics']['total_samples']}")
        print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"Top-3 Accuracy: {results['metrics']['top_3_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {results['metrics']['top_5_accuracy']:.4f}")
        print(f"Results saved to: {args.save_dir}")
        print("=" * 60)
        
        # Show best and worst performing classes
        if 'per_class_accuracy' in results:
            per_class_df = results['per_class_accuracy']
            
            print("\nTop 5 Best Performing Classes:")
            print(per_class_df.tail(5)[['class', 'accuracy']].to_string(index=False))
            
            print("\nTop 5 Worst Performing Classes:")
            print(per_class_df.head(5)[['class', 'accuracy']].to_string(index=False))
        
        # Show some statistics
        print(f"\nAverage per-class accuracy: {per_class_df['accuracy'].mean():.4f}")
        print(f"Standard deviation: {per_class_df['accuracy'].std():.4f}")
        print(f"Min accuracy: {per_class_df['accuracy'].min():.4f}")
        print(f"Max accuracy: {per_class_df['accuracy'].max():.4f}")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        return
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()