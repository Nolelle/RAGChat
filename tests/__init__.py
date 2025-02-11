"""Test suite for the RAG-based first responder chatbot.

This module provides test configurations and utilities that ensure consistent
test behavior across the entire test suite. It includes helpers for managing
test data, logging configuration, and common test assertions.
"""

import logging
import sys
from pathlib import Path
from typing import Any

# Add src directory to Python path for test imports
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

# Configure logging for tests - use a format that makes debugging easier
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.NullHandler()]
)

# Test data directory setup
TEST_DATA_DIR = Path(__file__).parent / "test_data"
SAMPLE_DOCS_DIR = TEST_DATA_DIR / "sample_documents"

# Create directories if they don't exist
TEST_DATA_DIR.mkdir(exist_ok=True)
SAMPLE_DOCS_DIR.mkdir(exist_ok=True)

# Common test data
SAMPLE_TEXTS = {
    'empty': '',
    'single_word': 'Test',
    'short': 'This is a short test document.',
    'special_chars': 'Text with @#$% special chars!',
    'multiple_spaces': 'Text  with   multiple    spaces',
}

def get_test_file_path(filename: str) -> Path:
    """Get the full path for a test file.

    Args:
        filename: Name of the test file

    Returns:
        Path object for the test file location
    """
    return TEST_DATA_DIR / filename

def create_test_document(content: str, filename: str) -> Path:
    """Create a test document with specified content.

    Args:
        content: Text content for the document
        filename: Name of the file to create

    Returns:
        Path object for the created file
    """
    file_path = SAMPLE_DOCS_DIR / filename
    file_path.write_text(content)
    return file_path

def clean_text_for_comparison(text: str) -> str:
    """Normalize text for consistent comparison in tests.

    Handles edge cases like multiple spaces and special characters.

    Args:
        text: Text to normalize

    Returns:
        Normalized text suitable for comparisons
    """
    import re
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    return text.strip()

def assert_texts_equal(actual: str, expected: str, msg: str = None):
    """Assert that two texts are equal after normalization.

    Args:
        actual: Actual text from the system
        expected: Expected text
        msg: Optional message to display on failure
    """
    actual_clean = clean_text_for_comparison(actual)
    expected_clean = clean_text_for_comparison(expected)
    assert actual_clean == expected_clean, (
        msg or f"\nExpected (cleaned): {expected_clean!r}"
        f"\nActual (cleaned): {actual_clean!r}"
    )

def setup_test_logger(name: str) -> logging.Logger:
    """Set up a logger for a test module.

    Args:
        name: Name for the logger

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger
