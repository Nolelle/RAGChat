#!/usr/bin/env python3
"""
Debug script for model response extraction.
This script helps identify issues with the response extraction from model output.
"""

import logging
import sys
import os
from pathlib import Path

# Add the src directory to the path for imports
sys.path.append(str(Path(__file__).parents[2] / "src"))

from firstresponders_chatbot.rag.rag_system import RAGSystem

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_extraction_debug.log"),
    ],
)
logger = logging.getLogger("debug_extract")


def test_extract_answer():
    """Test the extract answer function with different patterns."""
    # Initialize the RAG system
    logger.info("Initializing RAG system for testing")
    rag = RAGSystem()

    # Test cases: different formats of model output
    test_cases = [
        # Case 1: Ideal case with clear markers
        {
            "name": "clean_markers",
            "prompt": "<|system|>\nYou are a helpful assistant\n<|user|>\nWhat is first aid?\n<|assistant|>",
            "full_output": "<|system|>\nYou are a helpful assistant\n<|user|>\nWhat is first aid?\n<|assistant|>\nFirst aid is emergency care provided to an injured or sick person before professional medical help arrives.",
        },
        # Case 2: Missing end marker
        {
            "name": "missing_end_marker",
            "prompt": "<|system|>\nYou are a helpful assistant\n<|user|>\nWhat is first aid?\n<|assistant|>",
            "full_output": "<|system|>\nYou are a helpful assistant\n<|user|>\nWhat is first aid?\n<|assistant|>\nFirst aid is emergency care provided to an injured or sick person before professional medical help arrives.",
        },
        # Case 3: Multiple assistant markers
        {
            "name": "multiple_assistant_markers",
            "prompt": "<|system|>\nYou are a helpful assistant\n<|user|>\nWhat is first aid?\n<|assistant|>",
            "full_output": "<|system|>\nYou are a helpful assistant\n<|user|>\nWhat is first aid?\n<|assistant|>\nFirst aid is <|assistant|>\nemergency care provided to an injured or sick person before professional medical help arrives.",
        },
        # Case 4: No markers at all, just text
        {
            "name": "no_markers",
            "prompt": "<|system|>\nYou are a helpful assistant\n<|user|>\nWhat is first aid?\n<|assistant|>",
            "full_output": "First aid is emergency care provided to an injured or sick person before professional medical help arrives.",
        },
        # Case 5: Completely garbled output
        {
            "name": "garbled_output",
            "prompt": "<|system|>\nYou are a helpful assistant\n<|user|>\nWhat is first aid?\n<|assistant|>",
            "full_output": ":::...:...:;;:...:::::....:::::.::::..:.:.:.:....::.:.::::.::.:.:::.:",
        },
        # Case 6: Empty output
        {
            "name": "empty_output",
            "prompt": "<|system|>\nYou are a helpful assistant\n<|user|>\nWhat is first aid?\n<|assistant|>",
            "full_output": "",
        },
        # Case 7: Just repeated dots or other characters
        {
            "name": "repeated_chars",
            "prompt": "<|system|>\nYou are a helpful assistant\n<|user|>\nWhat is first aid?\n<|assistant|>",
            "full_output": "<|system|>\nYou are a helpful assistant\n<|user|>\nWhat is first aid?\n<|assistant|>\n..............................................................",
        },
        # Case 8: Invalid but realistic output
        {
            "name": "realistic_invalid",
            "prompt": "<|system|>\nYou are a helpful assistant\n<|user|>\nWhat is first aid?\n<|assistant|>",
            "full_output": "<|system|>\nYou are a helpful assistant\n<|user|>\nWhat is first aid?\n<|assistant|>\n: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :",
        },
    ]

    # Run tests
    for idx, case in enumerate(test_cases):
        logger.info(f"===== TESTING CASE {idx+1}: {case['name']} =====")

        try:
            # Test extraction
            result = rag._extract_answer_from_output(
                case["full_output"], case["prompt"]
            )

            # Log results
            logger.info(f"Extraction result: '{result}'")

            # Also test if the result would be considered garbled
            is_garbled = rag._detect_garbled_text(result, is_model_output=True)
            logger.info(f"Would be considered garbled: {is_garbled}")

            # Test the cleaning function
            cleaned = rag._clean_model_output(result)
            logger.info(f"After cleaning: '{cleaned}'")

        except Exception as e:
            logger.error(f"Error testing case {idx+1}: {str(e)}")

        logger.info(f"===== END OF CASE {idx+1} =====\n")


if __name__ == "__main__":
    logger.info("Starting debug script for response extraction")
    test_extract_answer()
    logger.info("Debug script completed")
