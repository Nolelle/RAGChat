#!/usr/bin/env python3
"""
Script to fix and flatten JSON dataset structure for Phi-3 training.
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


def fix_and_flatten_dataset(input_path, output_path):
    """Fix JSON dataset and flatten structure so input/output/text are directly accessible."""
    logger.info(f"Loading dataset from {input_path}")

    # Read the original file as text first
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        # Try to parse as JSON
        data = json.loads(content)
        logger.info("Successfully parsed JSON file")

        # Extract and flatten the train data
        if "train" in data and isinstance(data["train"], list):
            train_data = data["train"]
            logger.info(f"Found {len(train_data)} training examples")

            # Flatten the structure - create a list of items with input, output, text keys
            flattened_data = []
            for idx, item in enumerate(train_data):
                if (
                    isinstance(item, dict)
                    and "input" in item
                    and "output" in item
                    and "text" in item
                ):
                    flattened_data.append(
                        {
                            "input": item["input"],
                            "output": item["output"],
                            "text": item["text"],
                        }
                    )
                else:
                    logger.warning(
                        f"Item {idx} is missing required fields or has wrong structure"
                    )

            logger.info(
                f"Created flattened dataset with {len(flattened_data)} examples"
            )

            # Write the fixed and flattened data
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(flattened_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Fixed and flattened dataset saved to {output_path}")
            return True
        else:
            logger.error("Dataset doesn't have the expected 'train' key structure")
            return False

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        return False


def main():
    # Input and output files
    input_file = "data/phi3_dataset.json"
    output_file = "data/phi3_dataset_flattened.json"

    # Fix and flatten the dataset
    success = fix_and_flatten_dataset(input_file, output_file)

    if success:
        logger.info(
            "Dataset fixed and flattened successfully. Use the new file for training."
        )
        logger.info(f"Use: --dataset_path {output_file}")
    else:
        logger.error("Failed to fix and flatten the dataset.")


if __name__ == "__main__":
    main()
