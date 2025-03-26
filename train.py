#!/usr/bin/env python3
"""
Script to train the FirstRespondersChatbot model with Llama 2.
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
    # Check for Apple Silicon
    is_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    parser = argparse.ArgumentParser(
        description="Train the FirstRespondersChatbot model with Llama 2."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
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
        default="llama2-first-responder",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,  # Keep batch size at 1 due to memory constraints
        help="Batch size for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=32 if is_mps else 16,  # Higher gradient accumulation for MPS (32 vs 16)
        help="Number of steps to accumulate gradients",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048 if is_mps else 3072,  # Lower from 3072 to 2048 for Apple Silicon
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,  # Lowered from 2e-4 to 1e-4 for more stable learning
        help="Learning rate for training",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=2,  # Reduced to 2 epochs for Llama 2
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=is_mps,  # Enable by default for MPS
        help="Whether to use mixed precision training",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=not is_mps,  # Disable 4-bit quantization on MPS by default
        help="Whether to load model in 4-bit precision",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,  # Keep at 16
        help="Rank of the LoRA update matrices",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,  # Keep at 32
        help="Scaling factor for LoRA",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,  # Increased from 0.05 to 0.1 for better generalization
        help="Dropout probability for LoRA layers",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,  # Added warmup_ratio parameter
        help="Fraction of training steps for learning rate warmup",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,  # Added weight_decay parameter
        help="Weight decay for AdamW optimizer",
    )
    parser.add_argument(
        "--train_test_split",
        type=float,
        default=0.1,
        help="Fraction of data to use for testing",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of training samples to use (for faster training)",
    )
    # Apple Silicon specific arguments
    if is_mps:
        parser.add_argument(
            "--mps_memory_limit",
            type=int,
            default=None,
            help="Set the MPS memory limit (in GB) for training on Apple Silicon",
        )
        parser.add_argument(
            "--mps_enable_eager_mode",
            action="store_true",
            default=True,
            help="Enable eager mode for MPS, may improve stability",
        )

    args = parser.parse_args()

    # Apply Apple Silicon specific modifications to arguments if needed
    if is_mps:
        logger.info("Running on Apple Silicon (MPS) - applying optimized parameters")

        # Ensure batch size is 1 for MPS
        if args.batch_size > 1:
            logger.warning(
                f"Batch size {args.batch_size} is too high for MPS, reducing to 1"
            )
            args.batch_size = 1

        # Increase gradient accumulation if needed
        if args.gradient_accumulation_steps < 16:
            logger.info(
                f"Increasing gradient accumulation from {args.gradient_accumulation_steps} to 32 for MPS"
            )
            args.gradient_accumulation_steps = 32

        # Apply MPS optimizations if requested
        if hasattr(args, "mps_memory_limit") and args.mps_memory_limit:
            # Convert GB to bytes
            memory_limit = args.mps_memory_limit * 1024 * 1024 * 1024
            # Check if the function exists (some older versions don't have it)
            if hasattr(torch.backends.mps, "set_mem_quota"):
                logger.info(f"Setting MPS memory limit to {args.mps_memory_limit}GB")
                torch.backends.mps.set_mem_quota(memory_limit)

        # Enable eager mode if requested
        if hasattr(args, "mps_enable_eager_mode") and args.mps_enable_eager_mode:
            os.environ["PYTORCH_ENABLE_MPS_EAGER_FALLBACK"] = "1"
            logger.info("Enabled MPS eager fallback mode")

    return args


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

        # Clear MPS cache before starting
        torch.mps.empty_cache()

        # Set optimal thread count for Apple Silicon
        os.environ["OMP_NUM_THREADS"] = str(min(8, os.cpu_count() or 1))
        logger.info(
            f"Set optimal thread count for Apple Silicon: {os.environ.get('OMP_NUM_THREADS')}"
        )
    else:
        use_8bit = True

    # Load the dataset - expecting it to be preprocessed for Llama 2 already
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
    # For Apple Silicon, recommend smaller dataset if large
    elif is_mps and len(dataset) > 1000:
        logger.warning(
            f"Large dataset ({len(dataset)} examples) detected on Apple Silicon. "
            "Consider using --max_train_samples=500 if you encounter memory issues."
        )

    # Split the dataset into train and evaluation sets
    split_dataset = dataset.train_test_split(test_size=args.train_test_split, seed=42)
    logger.info(
        f"Split dataset into {len(split_dataset['train'])} train and {len(split_dataset['test'])} evaluation examples"
    )

    # Create model trainer with optimized parameters for Llama 2
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
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        model_format="llama2",  # Hardcoded to Llama 2
    )

    try:
        # Train the model
        logger.info("Starting Llama 2 fine-tuning...")
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
