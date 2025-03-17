#!/usr/bin/env python3
"""
Script to run the document preprocessor.
"""

import argparse
from .preprocessor import DocumentPreprocessor


def main():
    """Main function to run the document preprocessor."""
    parser = argparse.ArgumentParser(
        description="Preprocess documents for the FirstRespondersChatbot"
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="docs",
        help="Directory containing the documents to preprocess",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save the preprocessed documents",
    )
    args = parser.parse_args()

    preprocessor = DocumentPreprocessor(
        docs_dir=args.docs_dir, output_dir=args.output_dir
    )
    preprocessor.run()


if __name__ == "__main__":
    main()
