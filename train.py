#!/usr/bin/env python3
"""
Script to train the FirstRespondersChatbot model.
"""

import sys
from src.firstresponders_chatbot.training.trainer import ModelTrainer


def main():
    """Main function to train the model."""
    trainer = ModelTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
