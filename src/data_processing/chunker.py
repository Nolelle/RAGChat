from typing import List, Optional
import logging
from transformers import AutoTokenizer
from .types import TextChunk

logger = logging.getLogger(__name__)

class DocumentChunker:
    """Splits documents into semantic chunks while preserving context."""

    def __init__(self,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 model_name: str = "google/flan-t5-base"):
        """Initialize chunker with specified parameters.

        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            model_name: Name of the model whose tokenizer to use
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info(f"Initializing tokenizer with model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def create_chunks(self, text: str, source_doc: str) -> List[TextChunk]:
        """Split text into overlapping chunks.

        Args:
            text: Text content to split into chunks
            source_doc: Identifier for the source document

        Returns:
            List of TextChunk objects
        """
        # Tokenize the text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            # Calculate end index for current chunk
            end_idx = min(start_idx + self.chunk_size, len(tokens))

            # Get tokens for this chunk
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode tokens back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            # Clean up any leading/trailing whitespace
            chunk_text = chunk_text.strip()

            # Only create chunk if it contains text
            if chunk_text:
                chunk = TextChunk(
                    text=chunk_text,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    source_doc=source_doc
                )
                chunks.append(chunk)

                logger.debug(
                    f"Created chunk {len(chunks)} from {source_doc}: "
                    f"{len(chunk_tokens)} tokens"
                )

            # Move start index for next chunk, considering overlap
            start_idx = end_idx - self.chunk_overlap

            # If we're at the end and the remaining text is too small, stop
            if len(tokens) - start_idx < self.chunk_size // 2:
                break

        logger.info(
            f"Split document {source_doc} into {len(chunks)} chunks "
            f"(avg {sum(len(c.text) for c in chunks)/len(chunks):.1f} chars per chunk)"
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
            'num_chunks': len(chunks),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'total_length': sum(lengths)
        }
