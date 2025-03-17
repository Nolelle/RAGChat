#!/usr/bin/env python3
"""
Main script to run the FirstRespondersChatbot project.
"""

import sys
import argparse
from src.firstresponders_chatbot import __main__ as firstresponders_main


def main():
    """Main function to run the FirstRespondersChatbot project."""
    # Call the main function from the package
    firstresponders_main.main()


if __name__ == "__main__":
    main()
