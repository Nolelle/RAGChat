"""CLI module for RAG chatbot.

This module provides command-line interfaces for the chatbot.
"""

import logging

# Set up logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Import public interfaces
from .main import app

__all__ = [
    "app",
]

# Version of the cli module
__version__ = "0.1.0"
