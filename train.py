#!/usr/bin/env python3
"""
Script to train the FirstRespondersChatbot model.
"""

import sys
import argparse
from src.firstresponders_chatbot.training.trainer import ModelTrainer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the FirstRespondersChatbot model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/flan-t5-small",
        choices=[
            "google/flan-t5-small",
            "google/flan-t5-base",
            "google/flan-t5-large",
            "google/flan-t5-xl",
            "google/flan-t5-xxl",
        ],
        help="Name of the pre-trained model to use",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/pseudo_data.json",
        help="Path to the training dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flan-t5-first-responder",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of steps to accumulate gradients before performing an update",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=384,
        help="Maximum length of the source sequences",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=96,
        help="Maximum length of the target sequences",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use mixed precision training",
    )
    return parser.parse_args()


def main():
    """Main function to train the model."""
    args = parse_args()
    trainer = ModelTrainer(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
    )
    trainer.run()


if __name__ == "__main__":
    main()
