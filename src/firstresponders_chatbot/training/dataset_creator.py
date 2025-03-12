"""
Dataset creation module for the FirstRespondersChatbot project.

This module provides functionality to generate a pseudo-supervised dataset
for fine-tuning the Flan-T5-Small model.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetCreator:
    """Dataset creator for the FirstRespondersChatbot project."""

    def __init__(
        self,
        input_file: str = "data/preprocessed_data.json",
        output_file: str = "data/pseudo_data.json",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        """
        Initialize the dataset creator.

        Args:
            input_file: Path to the preprocessed data file
            output_file: Path to save the generated dataset
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            test_ratio: Ratio of data to use for testing
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Ensure ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        if not 0.999 <= total_ratio <= 1.001:  # Allow for floating point errors
            raise ValueError(f"Ratios must sum to 1, got {total_ratio}")

    def load_preprocessed_data(self) -> List[Dict[str, Any]]:
        """
        Load preprocessed data from JSON file.

        Returns:
            List of document dictionaries
        """
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        with open(self.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} documents from {self.input_file}")
        return data

    def generate_question_answer_pairs(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Generate question-answer pairs from documents.

        Args:
            documents: List of document dictionaries

        Returns:
            List of question-answer pairs
        """
        qa_pairs = []

        for doc in documents:
            content = doc["content"]
            if not content or len(content.strip()) < 50:
                continue  # Skip short or empty documents

            # Generate a simple question based on the content
            # In a real scenario, you might use more sophisticated methods
            qa_pair = {
                "question": f"What information can you provide about the following: {content[:100]}...",
                "answer": content,
            }
            qa_pairs.append(qa_pair)

        logger.info(f"Generated {len(qa_pairs)} question-answer pairs")
        return qa_pairs

    def split_data(
        self, qa_pairs: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Split data into training, validation, and test sets.

        Args:
            qa_pairs: List of question-answer pairs

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Shuffle data
        random.shuffle(qa_pairs)

        # Calculate split indices
        n_samples = len(qa_pairs)
        train_idx = int(n_samples * self.train_ratio)
        val_idx = train_idx + int(n_samples * self.val_ratio)

        # Split data
        train_data = qa_pairs[:train_idx]
        val_data = qa_pairs[train_idx:val_idx]
        test_data = qa_pairs[val_idx:]

        logger.info(
            f"Split data into {len(train_data)} training, {len(val_data)} validation, and {len(test_data)} test samples"
        )
        return train_data, val_data, test_data

    def format_for_flan_t5(
        self,
        train_data: List[Dict[str, str]],
        val_data: List[Dict[str, str]],
        test_data: List[Dict[str, str]],
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Format data for Flan-T5 fine-tuning.

        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data

        Returns:
            Dictionary with formatted data
        """
        # Format data for Flan-T5
        formatted_data = {
            "train": [
                {"input": f"question: {item['question']}", "output": item["answer"]}
                for item in train_data
            ],
            "validation": [
                {"input": f"question: {item['question']}", "output": item["answer"]}
                for item in val_data
            ],
            "test": [
                {"input": f"question: {item['question']}", "output": item["answer"]}
                for item in test_data
            ],
        }

        logger.info("Formatted data for Flan-T5 fine-tuning")
        return formatted_data

    def save_dataset(self, dataset: Dict[str, List[Dict[str, str]]]) -> None:
        """
        Save dataset to JSON file.

        Args:
            dataset: Dictionary with formatted data
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_file.parent, exist_ok=True)

        # Save dataset
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

        logger.info(f"Saved dataset to {self.output_file}")

    def run(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Run the dataset creation pipeline.

        Returns:
            Dictionary with formatted data
        """
        logger.info("Starting dataset creation...")

        # Load preprocessed data
        documents = self.load_preprocessed_data()

        # Generate question-answer pairs
        qa_pairs = self.generate_question_answer_pairs(documents)

        # Split data
        train_data, val_data, test_data = self.split_data(qa_pairs)

        # Format data for Flan-T5
        dataset = self.format_for_flan_t5(train_data, val_data, test_data)

        # Save dataset
        self.save_dataset(dataset)

        logger.info("Dataset creation completed successfully!")
        return dataset
