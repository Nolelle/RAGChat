#!/usr/bin/env python3
"""
Script to create a dataset for training the FirstRespondersChatbot.
"""

import sys
from src.firstresponders_chatbot.preprocessing.dataset_creator import DatasetCreator


def main():
    """Main function to create the dataset."""
    dataset_creator = DatasetCreator()
    dataset_creator.run()


if __name__ == "__main__":
    main()
