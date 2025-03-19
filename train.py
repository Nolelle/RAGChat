#!/usr/bin/env python3
"""
Script to train the FirstRespondersChatbot model.
"""

import sys
import argparse
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from src.firstresponders_chatbot.training.trainer import ModelTrainer
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Download NLTK resources
def download_nltk_resources():
    """Download required NLTK resources if not already present."""
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download("punkt")
        nltk.download("stopwords")
        print("NLTK resources downloaded.")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the FirstRespondersChatbot model with Phi-3 Mini."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Base model to use",
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
        default="phi-3-mini-first-responder",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of steps to accumulate gradients",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use mixed precision training",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Whether to load model in 4-bit precision",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="Rank of the LoRA update matrices",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="Scaling factor for LoRA",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout probability for LoRA layers",
    )
    parser.add_argument(
        "--train_test_split",
        type=float,
        default=0.1,
        help="Fraction of data to use for testing",
    )
    parser.add_argument(
        "--rebuild_dataset",
        action="store_true",
        help="Whether to rebuild the dataset from preprocessed documents",
    )
    parser.add_argument(
        "--skip_preprocessing",
        action="store_true",
        help="Skip the preprocessing step to use the raw dataset",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of training samples to use (for faster training)",
    )
    return parser.parse_args()


def main():
    """Main function to train the model."""
    # Download NLTK resources
    download_nltk_resources()

    args = parse_args()

    # Rebuild dataset if requested
    if args.rebuild_dataset:
        from src.firstresponders_chatbot.preprocessing.preprocessor import (
            DocumentPreprocessor,
        )
        from src.firstresponders_chatbot.training.dataset_creator import DatasetCreator

        print("Rebuilding dataset from documents...")
        # Preprocess documents
        preprocessor = DocumentPreprocessor()
        preprocessor.run()

        # Create dataset
        dataset_creator = DatasetCreator()
        dataset_creator.run()
        print("Dataset rebuilt successfully.")

    # Create model trainer
    trainer = ModelTrainer(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_seq_length=args.max_seq_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_train_samples=args.max_train_samples,
    )

    try:
        # Load the dataset
        dataset = trainer.load_dataset()

        # Log dataset information
        logger.info(f"Dataset loaded with columns: {dataset.column_names}")
        logger.info(f"Dataset size: {len(dataset)} examples")

        # Split the dataset if needed
        if args.train_test_split > 0:
            dataset = dataset.shuffle(seed=42)
            split = dataset.train_test_split(test_size=args.train_test_split)
            train_dataset = split["train"]
            eval_dataset = split["test"]
            logger.info(
                f"Split dataset into {len(train_dataset)} train and {len(eval_dataset)} evaluation examples"
            )
        else:
            train_dataset = dataset
            eval_dataset = None
            logger.info(f"Using all {len(train_dataset)} examples for training")

        # Apply formatting for Phi-3 instead of preprocessing
        if args.skip_preprocessing:
            logger.info("Skipping preprocessing as requested.")
        else:
            logger.info("Formatting dataset for Phi-3")
            train_dataset = train_dataset
            if eval_dataset is not None:
                eval_dataset = eval_dataset

        # Train the model
        trainer.train(train_dataset=train_dataset, eval_dataset=eval_dataset)
        logger.info(f"Model trained and saved to {args.output_dir}")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
