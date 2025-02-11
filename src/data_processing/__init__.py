"""Data processing module for RAG chatbot.

This module handles document processing, including:
- Converting various document formats to clean text
- Chunking text into semantic units for embedding
- Managing document metadata
"""

import logging

# Set up logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Import public interfaces
from .types import TextChunk
from .converter import DocumentConverter
from .chunker import DocumentChunker
from .processor import DocumentProcessor

__all__ = [
    'TextChunk',
    'DocumentConverter',
    'DocumentChunker',
    'DocumentProcessor',
]

# Version of the data_processing module
__version__ = '0.1.0'
