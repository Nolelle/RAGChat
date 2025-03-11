#!/usr/bin/env python3
"""
create_dataset.py - Dataset creation script for the FirstRespondersChatbot project.

This script takes preprocessed text chunks from preprocess.py and creates
input-output pairs for fine-tuning the Flan-T5-Small model, saving them as a
JSON file suitable for training.
"""

import json
import random
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File paths
DATA_DIR = Path("data")
PREPROCESSED_DATA_PATH = DATA_DIR / "preprocessed_data.json"
DATASET_PATH = DATA_DIR / "pseudo_data.json"

# Templates for generating questions
QUESTION_TEMPLATES = [
    "What should I do if {}?",
    "How do I handle a situation where {}?",
    "What is the proper procedure for {}?",
    "How should first responders deal with {}?",
    "What are the steps for {}?",
    "As a first responder, what should I know about {}?",
    "What is the protocol for {}?",
    "Can you explain how to {}?",
    "What's the best way to respond to {}?",
    "How would you manage {}?",
]

# Keywords to extract potential topics from chunks
TOPIC_INDICATORS = [
    "protocol",
    "procedure",
    "emergency",
    "response",
    "steps",
    "guideline",
    "instruction",
    "safety",
    "treatment",
    "handling",
    "action",
    "precaution",
    "measure",
    "technique",
    "method",
]


def load_preprocessed_data() -> List[Dict[str, Any]]:
    """
    Load preprocessed data from JSON file.

    Returns:
        List[Dict[str, Any]]: List of document dictionaries.
    """
    if not PREPROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessed data file not found: {PREPROCESSED_DATA_PATH}"
        )

    with open(PREPROCESSED_DATA_PATH, "r") as f:
        documents = json.load(f)

    logger.info(f"Loaded {len(documents)} documents from {PREPROCESSED_DATA_PATH}")
    return documents


def extract_topic_from_chunk(chunk: str) -> str:
    """
    Extract a potential topic from a text chunk for question generation.

    Args:
        chunk: Text content to extract topic from.

    Returns:
        str: Extracted topic or empty string if none found.
    """
    # Try to find a heading-like pattern (e.g., "## CPR Protocol")
    heading_match = re.search(r"#{1,3}\s*(.*?)(?:\n|$)", chunk)
    if heading_match:
        return heading_match.group(1).strip()

    # Look for sentences containing topic indicators
    for indicator in TOPIC_INDICATORS:
        pattern = rf"([^.!?\n]*{indicator}[^.!?\n]*[.!?])"
        matches = re.findall(pattern, chunk, re.IGNORECASE)
        if matches:
            # Return the first match, removing the period
            return re.sub(r"[.!?]$", "", matches[0].strip())

    # If no good topic is found, take the first sentence
    first_sentence = re.search(r"^[^.!?\n]*[.!?]", chunk)
    if first_sentence:
        return re.sub(r"[.!?]$", "", first_sentence.group(0).strip())

    return ""


def generate_question_answer_pair(chunk: str) -> Tuple[str, str]:
    """
    Generate a question-answer pair from a text chunk.

    Args:
        chunk: Text content to generate question-answer from.

    Returns:
        Tuple[str, str]: Question and answer pair.
    """
    topic = extract_topic_from_chunk(chunk)

    if not topic:
        # If no topic found, use a generic question template
        question = "What information can you provide about first responder protocols?"
    else:
        # Generate question from template
        template = random.choice(QUESTION_TEMPLATES)
        question = template.format(topic.lower())

    # The answer is the original text chunk
    answer = chunk

    return question, answer


def create_dataset(documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Create a dataset of question-answer pairs from preprocessed documents.

    Args:
        documents: List of preprocessed document dictionaries.

    Returns:
        List[Dict[str, str]]: List of dictionaries with input and target fields.
    """
    dataset = []

    for doc in documents:
        chunk = doc["content"]

        # Skip very short chunks
        if len(chunk.split()) < 10:
            continue

        # Generate question-answer pair
        question, answer = generate_question_answer_pair(chunk)

        # Format for T5 fine-tuning
        # T5 expects "question: <question>" as input
        # and "<answer>" as target
        input_text = f"question: {question}"
        target_text = answer

        dataset.append({"input": input_text, "target": target_text})

    logger.info(f"Created {len(dataset)} question-answer pairs")
    return dataset


def save_dataset(dataset: List[Dict[str, str]]) -> None:
    """
    Save dataset to a JSON file.

    Args:
        dataset: List of dictionaries with input and target fields.
    """
    # Create output directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)

    with open(DATASET_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    logger.info(f"Saved dataset with {len(dataset)} examples to {DATASET_PATH}")


def main():
    """Main function to create the dataset."""
    logger.info("Starting dataset creation...")

    # Load preprocessed data
    documents = load_preprocessed_data()

    # Create dataset
    dataset = create_dataset(documents)

    # Save dataset
    save_dataset(dataset)

    logger.info("Dataset creation completed successfully!")


if __name__ == "__main__":
    main()
