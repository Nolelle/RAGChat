from pathlib import Path
from typing import List, Optional
import logging
from .converter import DocumentConverter
from .chunker import DocumentChunker
from .types import TextChunk

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Main class that combines conversion and chunking functionality."""

    def __init__(self,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 model_name: str = "google/flan-t5-base"):
        """Initialize processor with converter and chunker.

        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            model_name: Name of the model whose tokenizer to use
        """
        logger.info("Initializing DocumentProcessor")
        self.converter = DocumentConverter()
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name=model_name
        )

    def process_document(self, file_path: Path) -> List[TextChunk]:
        """Process a document from file to chunks.

        Args:
            file_path: Path to the document file

        Returns:
            List of TextChunk objects

        Raises:
            FileNotFoundError: If the document doesn't exist
            ValueError: If the file format is not supported
        """
        logger.info(f"Processing document: {file_path}")

        # Convert document to text
        text = self.converter.load_document(file_path)

        # Create chunks from text
        chunks = self.chunker.create_chunks(text, file_path.name)

        # Get and log statistics
        stats = self.chunker.get_chunk_stats(chunks)
        logger.info(
            f"Document processing complete. Stats: "
            f"{stats['num_chunks']} chunks, "
            f"{stats['avg_length']:.1f} avg chars per chunk"
        )

        return chunks

    def process_text(self, text: str, source_identifier: str) -> List[TextChunk]:
        """Process text directly without loading from file.

        Args:
            text: Text content to process
            source_identifier: Identifier for the source

        Returns:
            List of TextChunk objects
        """
        logger.info(f"Processing text from source: {source_identifier}")

        # Clean the text
        cleaned_text = self.converter.convert_text(text, source_identifier)

        # Create chunks
        chunks = self.chunker.create_chunks(cleaned_text, source_identifier)

        return chunks
