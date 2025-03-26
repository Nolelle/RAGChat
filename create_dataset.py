#!/usr/bin/env python3
"""
Script to create a dataset for training the FirstRespondersChatbot with Llama 2.
"""

import sys
import argparse
import logging
from src.firstresponders_chatbot.training.dataset_creator import DatasetCreator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a dataset for training the FirstRespondersChatbot with Llama 2."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/preprocessed_data.json",
        help="Path to the preprocessed data file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/pseudo_data.json",
        help="Path to save the generated dataset",
    )
    return parser.parse_args()


def main():
    """Create a dataset formatted for Llama 2."""
    args = parse_args()

    logger.info(f"Creating dataset with Llama 2 format from {args.input_file}")

    # Create dataset creator for Llama 2
    dataset_creator = DatasetCreator(
        input_file=args.input_file,
        output_file=args.output_file,
        model_format="llama2",  # Hardcoded to llama2
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        similarity_threshold=0.65,  # Optimized for Llama 2
        max_examples_per_doc=5,  # Generate more examples per document
    )

    # Run dataset creation
    dataset_creator.run()

    logger.info(f"Dataset successfully created and saved to {args.output_file}")
    logger.info("Dataset is formatted for LLAMA2 model")


if __name__ == "__main__":
    main()
