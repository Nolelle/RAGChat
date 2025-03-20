#!/usr/bin/env python3
"""
Script to create a dataset for training the FirstRespondersChatbot with Phi-3.
"""

import sys
import argparse
from src.firstresponders_chatbot.training.dataset_creator import DatasetCreator


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a dataset for training the FirstRespondersChatbot."
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
    parser.add_argument(
        "--model_format",
        type=str,
        default="phi-3",
        choices=["phi-3", "flan-t5", "llama"],
        help="Format to use for the dataset (phi-3, flan-t5, or llama)",
    )
    return parser.parse_args()


def main():
    """Main function to create the dataset."""
    args = parse_args()

    # Create dataset creator with specified format
    dataset_creator = DatasetCreator(
        input_file=args.input_file,
        output_file=args.output_file,
        model_format=args.model_format,
    )

    # Run dataset creation
    dataset_creator.run()

    print(f"Dataset created successfully and saved to {args.output_file}")
    print(f"Dataset is formatted for {args.model_format} model")


if __name__ == "__main__":
    main()
