#!/usr/bin/env python3
"""
Script to train the FirstRespondersChatbot model with Llama 3.1 1B.
"""

import sys
import argparse
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import logging
import os
import torch
from datasets import load_dataset
import json
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the trainer module
try:
    from src.firstresponders_chatbot.training.trainer import ModelTrainer
except ImportError:
    logger.error("Could not import ModelTrainer. Is the package installed?")
    sys.exit(1)


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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the FirstRespondersChatbot model with Llama 3.1 1B."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-1B",
        help="Base model to use (default: Llama 3.1 1B)",
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
        default="llama-3.1-1b-first-responder",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training (larger for Llama 3.1 1B)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of steps to accumulate gradients",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
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
        default=8,
        help="Rank of the LoRA update matrices",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
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


def load_and_check_json(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file and check its structure.

    Args:
        file_path: Path to the JSON file

    Returns:
        The loaded JSON data
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check if the file exists
        if not os.path.exists(file_path):
            logger.error(f"Dataset file not found: {file_path}")
            sys.exit(1)

        return data
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON in {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)


def main():
    """Main function to train the model."""
    # Download NLTK resources
    download_nltk_resources()

    args = parse_args()

    # Check for Apple Silicon and disable 8-bit optimization if needed
    is_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if is_mps:
        logger.info("Apple Silicon detected - disabling 8-bit optimization")
        use_8bit = False
    else:
        use_8bit = True

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

        # Create dataset specifically for Llama format
        dataset_creator = DatasetCreator(model_format="llama")
        dataset_creator.run()
        print("Dataset rebuilt successfully with Llama formatting.")

    # Load the dataset
    logger.info(f"Loading dataset from {args.dataset_path}")
    if args.dataset_path.endswith(".json"):
        dataset = load_dataset("json", data_files=args.dataset_path)
    elif args.dataset_path.endswith(".csv"):
        dataset = load_dataset("csv", data_files=args.dataset_path)
    else:
        try:
            # Try to load as a HuggingFace dataset
            dataset = load_dataset(args.dataset_path)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    # Check if the dataset has a 'train' split
    if "train" in dataset:
        logger.info("Found nested structure with 'train' key")
        dataset = dataset["train"]

    logger.info(f"Dataset loaded with columns: {dataset.column_names}")
    logger.info(f"Dataset size: {len(dataset)} examples")

    # If max_train_samples is specified, select a subset of the dataset
    if args.max_train_samples is not None and args.max_train_samples < len(dataset):
        logger.info(f"Using {args.max_train_samples} examples for training")
        dataset = dataset.select(range(args.max_train_samples))

    # Split the dataset into train and evaluation sets
    split_dataset = dataset.train_test_split(test_size=args.train_test_split, seed=42)
    logger.info(
        f"Split dataset into {len(split_dataset['train'])} train and {len(split_dataset['test'])} evaluation examples"
    )

    # For Llama 3.1 models, we use the llama specific formatting
    logger.info("Formatting dataset for Llama 3.1")
    # The Llama specific formatting will be handled in the trainer

    # Create model trainer
    trainer = ModelTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        use_8bit_optimizer=use_8bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    try:
        # Train the model
        trainer.train(
            train_dataset=split_dataset["train"], eval_dataset=split_dataset["test"]
        )
        logger.info(f"Model trained and saved to {args.output_dir}")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
