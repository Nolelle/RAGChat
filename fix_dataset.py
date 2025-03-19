#!/usr/bin/env python3
"""
Script to fix JSON dataset structure issues.
"""

import json
import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fix_phi3_dataset(input_path, output_path):
    """Fix JSON dataset structure issues."""
    logger.info(f"Loading dataset from {input_path}")

    # Read the original file as text first to avoid json parsing errors
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        # Try to parse as JSON
        data = json.loads(content)
        logger.info("Successfully parsed JSON file")

        # Check if it has the expected structure with 'train' key
        if "train" in data and isinstance(data["train"], list):
            train_data = data["train"]
            logger.info(f"Found {len(train_data)} training examples")

            # Validate and ensure all entries have the same structure
            fixed_train_data = []
            for idx, item in enumerate(train_data):
                # Ensure all required fields exist
                if "input" in item and "output" in item and "text" in item:
                    # Add to fixed data
                    fixed_train_data.append(
                        {
                            "input": item["input"],
                            "output": item["output"],
                            "text": item["text"],
                        }
                    )
                else:
                    logger.warning(f"Item {idx} is missing required fields, skipping")

            # Create a new dataset with consistent structure
            fixed_data = {"train": fixed_train_data}

            # Write the fixed data
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(fixed_data, f, ensure_ascii=False, indent=2)

            logger.info(
                f"Fixed dataset saved to {output_path} with {len(fixed_train_data)} examples"
            )
            return True
        else:
            logger.error("Dataset doesn't have the expected 'train' key structure")
            return False

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")

        # Try a more manual approach - read line by line and fix problems
        logger.info("Attempting manual fix by extracting entries...")

        # Create a simpler structure with key components
        fixed_train_data = []

        # Basic pattern detection for entries
        current_item = None
        lines = content.split("\n")

        for line in lines:
            line = line.strip()
            if '"input":' in line and current_item is None:
                # Start of a new item
                current_item = {"input": "", "output": "", "text": ""}
                # Extract the input part
                input_part = line.split('"input":')[1].strip()
                if input_part.startswith('"'):
                    # Remove leading quotes
                    input_part = input_part[1:]
                    # Find the end of the input string
                    current_item["input"] = input_part
            elif '"output":' in line and current_item is not None:
                # Extract the output part
                output_part = line.split('"output":')[1].strip()
                if output_part.startswith('"'):
                    # Remove leading quotes
                    output_part = output_part[1:]
                    # Find the end of the output string
                    current_item["output"] = output_part
            elif '"text":' in line and current_item is not None:
                # Extract the text part
                text_part = line.split('"text":')[1].strip()
                if text_part.startswith('"'):
                    # Remove leading quotes
                    text_part = text_part[1:]
                    # Find the end of the text string
                    current_item["text"] = text_part

                    # Add completed item to the list
                    if all(current_item.values()):
                        fixed_train_data.append(current_item)
                    current_item = None

        if fixed_train_data:
            # Create a new dataset with consistent structure
            fixed_data = {"train": fixed_train_data}

            # Write the fixed data
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(fixed_data, f, ensure_ascii=False, indent=2)

            logger.info(
                f"Manually fixed dataset saved to {output_path} with {len(fixed_train_data)} examples"
            )
            return True
        else:
            logger.error("Failed to manually fix the dataset")
            return False


def main():
    # Input and output files
    input_file = "data/phi3_dataset.json"
    output_file = "data/phi3_dataset_fixed.json"

    # Fix the dataset
    success = fix_phi3_dataset(input_file, output_file)

    if success:
        logger.info("Dataset fixed successfully. Use the new file for training.")
        logger.info(f"Use: --dataset_path {output_file}")
    else:
        logger.error("Failed to fix the dataset. Attempting alternative approach...")

        # Alternative approach: Convert to JSONL format
        output_jsonl = "data/phi3_dataset.jsonl"
        logger.info(f"Converting to JSONL format: {output_jsonl}")

        try:
            # Read the original file as text
            with open(input_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Try to extract { ... } blocks
            import re

            pattern = r"\{[^{}]*\}"
            matches = re.findall(pattern, content)

            if matches:
                with open(output_jsonl, "w", encoding="utf-8") as f:
                    for match in matches:
                        try:
                            # Try to parse each match as JSON
                            item = json.loads(match)
                            if "input" in item and "output" in item and "text" in item:
                                # Write valid items as JSONL
                                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        except:
                            # Skip invalid matches
                            pass

                logger.info(f"Created JSONL file: {output_jsonl}")
                logger.info(f"Use: --dataset_path {output_jsonl}")
            else:
                logger.error("No valid JSON blocks found")
        except Exception as e:
            logger.error(f"Alternative approach failed: {e}")


if __name__ == "__main__":
    main()
