#!/usr/bin/env python3
"""
Script to run the document preprocessor.
"""

import sys
from src.firstresponders_chatbot.preprocessing.preprocessor import DocumentPreprocessor


def main():
    """Main function to run the document preprocessor."""
    preprocessor = DocumentPreprocessor()
    preprocessor.run()


if __name__ == "__main__":
    main()
