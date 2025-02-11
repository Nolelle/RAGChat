from pathlib import Path
import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DocumentConverter:
    """Handles conversion of documents to clean text format."""

    def __init__(self):
        """Initialize the document converter."""
        self.supported_formats = {'.txt', '.pdf'}

    def convert_text(self, text: str, source: str) -> str:
        """Convert raw text to cleaned format.

        Args:
            text: Raw text content to clean
            source: Identifier for the source document

        Returns:
            Cleaned text with normalized whitespace and punctuation
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)

        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove multiple consecutive line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)

        logger.debug(f"Converted text from {source}: {len(text)} characters")
        return text

    def load_document(self, file_path: Path) -> str:
        """Load document from file path.

        Args:
            file_path: Path to the document file

        Returns:
            Cleaned text content of the document

        Raises:
            FileNotFoundError: If the document doesn't exist
            ValueError: If the file format is not supported
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in self.supported_formats:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: {self.supported_formats}"
            )

        if suffix == '.txt':
            return self._load_text(file_path)
        elif suffix == '.pdf':
            return self._load_pdf(file_path)

    def _load_text(self, file_path: Path) -> str:
        """Load and clean text file.

        Args:
            file_path: Path to the text file

        Returns:
            Cleaned text content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.convert_text(text, file_path.name)
        except UnicodeDecodeError:
            # Try alternative encodings if UTF-8 fails
            encodings = ['latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    logger.info(f"Successfully read file using {encoding} encoding")
                    return self.convert_text(text, file_path.name)
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode file {file_path} with any supported encoding")

    def _load_pdf(self, file_path: Path) -> str:
        """Load and convert PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted and cleaned text content

        Raises:
            NotImplementedError: PDF support not yet implemented
        """
        # TODO: Implement PDF loading using pdfplumber or similar
        raise NotImplementedError("PDF loading not yet implemented")
