#!/usr/bin/env python3
"""
Script to preprocess documents for the FirstRespondersChatbot.
"""

import sys
import argparse
import logging
from src.firstresponders_chatbot.preprocessing.preprocessor import DocumentPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess documents for the FirstRespondersChatbot."
    )
    parser.add_argument(
        "--docs_dir",
        type=str,
        default="docs",
        help="Directory containing the documents to preprocess",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save the preprocessed documents",
    )
    return parser.parse_args()


def main():
    """Run the document preprocessor to prepare data for Llama 3 training."""
    args = parse_args()

    logger.info(f"Starting document preprocessing from {args.docs_dir}")

    # Initialize and run the document preprocessor
    preprocessor = DocumentPreprocessor(
        docs_dir=args.docs_dir,
        output_dir=args.output_dir,
        batch_size=1,  # Use small batch size for Apple Silicon
    )

    # Run preprocessing
    preprocessor.run()

    logger.info(
        f"Preprocessing completed. Results saved to {args.output_dir}/preprocessed_data.json"
    )


if __name__ == "__main__":
    main()
