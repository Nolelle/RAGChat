#!/usr/bin/env python3
"""
Script to train the FirstRespondersChatbot model.
"""

import sys
import argparse
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from src.firstresponders_chatbot.training.trainer import ModelTrainer
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Download NLTK resources
def download_nltk_resources():
    """Download required NLTK resources if not already present."""
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download("punkt")
        nltk.download("stopwords")
        print("NLTK resources downloaded.")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the FirstRespondersChatbot model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/flan-t5-small",
        choices=[
            "google/flan-t5-small",
            "google/flan-t5-base",
            "google/flan-t5-large",
            "google/flan-t5-xl",
            "google/flan-t5-xxl",
        ],
        help="Name of the pre-trained model to use",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/pseudo_data.json",
        help="Path to the training dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flan-t5-first-responder",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=32,
        help="Number of steps to accumulate gradients before performing an update",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=384,
        help="Maximum length of the source sequences",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=96,
        help="Maximum length of the target sequences",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=8,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use mixed precision training",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Whether to freeze the encoder layers",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Whether to load model in 8-bit precision",
    )
    parser.add_argument(
        "--train_test_split",
        type=float,
        default=0.1,
        help="Fraction of data to use for testing",
    )
    parser.add_argument(
        "--rebuild_dataset",
        action="store_true",
        help="Whether to rebuild the dataset from preprocessed documents",
    )
    parser.add_argument(
        "--skip_preprocessing",
        action="store_true",
        help="Skip the preprocessing step to use the raw dataset",
    )
    return parser.parse_args()


def main():
    """Main function to train the model."""
    # Download NLTK resources
    download_nltk_resources()

    args = parse_args()

    # Rebuild dataset if requested
    if args.rebuild_dataset:
        from src.firstresponders_chatbot.preprocessing.preprocessor import (
            DocumentPreprocessor,
        )
        from src.firstresponders_chatbot.training.dataset_creator import DatasetCreator

        print("Rebuilding dataset from documents...")
        # Preprocess documents
        preprocessor = DocumentPreprocessor()
        preprocessor.run()

        # Create dataset
        dataset_creator = DatasetCreator()
        dataset_creator.run()
        print("Dataset rebuilt successfully.")

    # Create model trainer
    trainer = ModelTrainer(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        freeze_encoder=args.freeze_encoder,
        load_in_8bit=args.load_in_8bit,
    )

    try:
        # Load the dataset
        dataset = trainer.load_dataset()

        # Log dataset information
        logger.info(f"Dataset loaded with columns: {dataset.column_names}")
        logger.info(f"Dataset size: {len(dataset)} examples")

        # Split the dataset if needed
        if args.train_test_split > 0:
            dataset = dataset.shuffle(seed=42)
            split = dataset.train_test_split(test_size=args.train_test_split)
            train_dataset = split["train"]
            eval_dataset = split["test"]
            logger.info(
                f"Split dataset into {len(train_dataset)} train and {len(eval_dataset)} evaluation examples"
            )
        else:
            train_dataset = dataset
            eval_dataset = None
            logger.info(f"Using all {len(train_dataset)} examples for training")

        # Apply preprocessing if not skipped
        if args.skip_preprocessing:
            logger.info("Skipping preprocessing as requested.")
        else:
            logger.info("Applying preprocessing to dataset")
            try:
                train_dataset = trainer.preprocess_data(train_dataset)
                if eval_dataset is not None:
                    eval_dataset = trainer.preprocess_data(eval_dataset)
            except Exception as e:
                logger.error(f"Error during preprocessing: {e}")
                logger.info(
                    "Falling back to automatic format conversion during training"
                )

        # Train the model
        trainer.train(train_dataset=train_dataset, eval_dataset=eval_dataset)
        logger.info(f"Model trained and saved to {args.output_dir}")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
