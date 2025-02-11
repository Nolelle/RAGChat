from dataclasses import dataclass
from typing import Optional

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata.

    Attributes:
        text: The actual text content of the chunk
        start_idx: Starting index in the original document (token position)
        end_idx: Ending index in the original document (token position)
        source_doc: Name or identifier of the source document
    """
    text: str
    start_idx: int
    end_idx: int
    source_doc: str

    def __len__(self) -> int:
        """Return the length of the chunk text."""
        return len(self.text)

    def __str__(self) -> str:
        """String representation of the chunk."""
        return f"TextChunk(source='{self.source_doc}', text='{self.text[:50]}...')"
