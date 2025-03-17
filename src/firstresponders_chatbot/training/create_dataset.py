#!/usr/bin/env python3
"""
Script to run the dataset creator.
"""

import argparse
from .dataset_creator import DatasetCreator


def main():
    """Main function to run the dataset creator."""
    parser = argparse.ArgumentParser(
        description="Create a dataset for the FirstRespondersChatbot"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/preprocessed_data.json",
        help="Path to the preprocessed data file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/pseudo_data.json",
        help="Path to save the generated dataset",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio of data to use for validation",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Ratio of data to use for testing"
    )
    args = parser.parse_args()

    creator = DatasetCreator(
        input_file=args.input_file,
        output_file=args.output_file,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    creator.run()


if __name__ == "__main__":
    main()
