#!/usr/bin/env python3
"""
Script to rebuild the dataset with improved processing techniques.
"""

import argparse
import os
import nltk
import logging
from pathlib import Path
from src.firstresponders_chatbot.preprocessing.preprocessor import DocumentPreprocessor
from src.firstresponders_chatbot.training.dataset_creator import DatasetCreator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_nltk_resources():
    """Download required NLTK resources if not already present."""
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
    except LookupError:
        logger.info("Downloading NLTK resources...")
        nltk.download("punkt")
        nltk.download("stopwords")
        logger.info("NLTK resources downloaded.")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rebuild the dataset with improved processing techniques."
    )
    parser.add_argument(
        "--docs_dir",
        type=str,
        default="docs",
        help="Directory containing documents to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--min_content_length",
        type=int,
        default=100,
        help="Minimum length of content to consider",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.8,
        help="Threshold for detecting similar documents",
    )
    parser.add_argument(
        "--max_examples_per_doc",
        type=int,
        default=3,
        help="Maximum examples to generate from a document",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of data to use for validation",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Ratio of data to use for testing",
    )
    return parser.parse_args()


def main():
    """Main function to rebuild the dataset."""
    # Download NLTK resources
    download_nltk_resources()

    # Parse arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Preprocess documents
    logger.info(f"Preprocessing documents from {args.docs_dir}...")
    preprocessor = DocumentPreprocessor(
        docs_dir=args.docs_dir, output_dir=args.output_dir
    )
    documents = preprocessor.preprocess_documents()
    preprocessor.save_documents(documents)
    logger.info(f"Preprocessed {len(documents)} documents")

    # Step 2: Create dataset
    logger.info("Creating dataset...")
    dataset_creator = DatasetCreator(
        input_file=os.path.join(args.output_dir, "preprocessed_data.json"),
        output_file=os.path.join(args.output_dir, "pseudo_data.json"),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_content_length=args.min_content_length,
        similarity_threshold=args.similarity_threshold,
        max_examples_per_doc=args.max_examples_per_doc,
    )
    qa_pairs = dataset_creator.generate_question_answer_pairs(
        dataset_creator.load_preprocessed_data()
    )
    train_data, val_data, test_data = dataset_creator.split_data(qa_pairs)
    dataset_creator.save_data(train_data, val_data, test_data)

    logger.info(f"Created dataset with:")
    logger.info(f"  - {len(train_data)} training examples")
    logger.info(f"  - {len(val_data)} validation examples")
    logger.info(f"  - {len(test_data)} test examples")
    logger.info(f"Dataset saved to {os.path.join(args.output_dir, 'pseudo_data.json')}")


if __name__ == "__main__":
    main()
