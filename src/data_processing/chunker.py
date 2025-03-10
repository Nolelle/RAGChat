from typing import List, Dict, Any
import logging
from haystack.components.preprocessors import DocumentSplitter
from haystack import Document
from .types import TextChunk

logger = logging.getLogger(__name__)


class HaystackChunker:
    """Splits documents into semantic chunks using Haystack's DocumentSplitter."""

    def __init__(
        self, chunk_size: int = 500, chunk_overlap: int = 50, split_by: str = "sentence"
    ):
        """Initialize chunker with specified parameters.

        Args:
            chunk_size: Maximum size of chunks in characters
            chunk_overlap: Overlap between chunks in characters
            split_by: Method to split text ('sentence', 'word', 'passage', or 'character')
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_by = split_by

        logger.info(
            f"Initializing Haystack chunker with size: {chunk_size}, overlap: {chunk_overlap}"
        )
        self.splitter = DocumentSplitter(
            split_by=split_by, split_length=chunk_size, split_overlap=chunk_overlap
        )

    def create_chunks(self, text: str, source_doc: str) -> List[TextChunk]:
        """Split text into overlapping chunks.

        Args:
            text: Text content to split into chunks
            source_doc: Identifier for the source document

        Returns:
            List of TextChunk objects
        """
        # Create a Haystack Document
        document = Document(content=text, meta={"source": source_doc})

        # Split the document
        result = self.splitter.run(documents=[document])
        haystack_chunks = result["documents"]

        # Convert Haystack Documents to TextChunk objects
        chunks = []
        for i, doc in enumerate(haystack_chunks):
            # Use index positions as start/end since we don't have token positions
            chunk = TextChunk(
                text=doc.content,
                start_idx=i,  # Using index as a proxy for position
                end_idx=i + 1,  # Using index+1 as a proxy for end position
                source_doc=source_doc,
            )
            chunks.append(chunk)

        logger.info(
            f"Split document {source_doc} into {len(chunks)} chunks "
            f"(avg {sum(len(c.text) for c in chunks)/max(1, len(chunks)):.1f} chars per chunk)"
        )

        return chunks

    def get_chunk_stats(self, chunks: List[TextChunk]) -> dict:
        """Calculate statistics about the chunks.

        Args:
            chunks: List of TextChunk objects

        Returns:
            Dictionary containing statistics about the chunks
        """
        if not chunks:
            return {}

        lengths = [len(chunk.text) for chunk in chunks]
        return {
            "num_chunks": len(chunks),
            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "total_length": sum(lengths),
        }
