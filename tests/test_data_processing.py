import pytest
from pathlib import Path
from src.data_processing import TextChunk, DocumentConverter, DocumentChunker, DocumentProcessor

# Import our test utilities
from . import assert_texts_equal, SAMPLE_TEXTS, clean_text_for_comparison

# Fixtures
@pytest.fixture
def sample_text():
    """Provide a sample document text with various features to test."""
    return """This is a sample document.
    It has multiple lines and some special characters: @#$%
    We want to test how well it's processed.

    Even with multiple paragraphs.
    """

@pytest.fixture
def temp_text_file(tmp_path, sample_text):
    """Create a temporary text file with sample content."""
    file_path = tmp_path / "test.txt"
    file_path.write_text(sample_text)
    return file_path

@pytest.fixture
def converter():
    """Provide a fresh DocumentConverter instance."""
    return DocumentConverter()

@pytest.fixture
def chunker():
    """Provide a DocumentChunker with standard test settings."""
    return DocumentChunker(chunk_size=100, chunk_overlap=20)

@pytest.fixture
def processor():
    """Provide a DocumentProcessor with standard test settings."""
    return DocumentProcessor(chunk_size=100, chunk_overlap=20)

# TextChunk Tests
class TestTextChunk:
    def test_chunk_creation(self):
        """Test basic TextChunk creation and attribute access."""
        chunk = TextChunk(
            text="Test text",
            start_idx=0,
            end_idx=10,
            source_doc="test.txt"
        )
        assert chunk.text == "Test text"
        assert chunk.start_idx == 0
        assert chunk.end_idx == 10
        assert chunk.source_doc == "test.txt"

    def test_chunk_length(self):
        """Test that chunk length matches text length."""
        chunk = TextChunk("Hello world", 0, 11, "test.txt")
        assert len(chunk) == len("Hello world")

    def test_chunk_string_representation(self):
        """Test the string representation of TextChunk."""
        chunk = TextChunk("A very long text that should be truncated", 0, 40, "test.txt")
        str_rep = str(chunk)
        assert str_rep.startswith("TextChunk(source='test.txt', text='A very")
        assert '...' in str_rep

# DocumentConverter Tests
class TestDocumentConverter:
    def test_convert_text_basic(self, converter):
        """Test basic text conversion with space normalization."""
        text = "This is a  test   with extra   spaces"
        result = converter.convert_text(text, "test")
        assert_texts_equal(result, "This is a test with extra spaces")

    def test_convert_text_special_chars(self, converter):
        """Test text conversion with special characters."""
        text = "Text with @#$% special chars!"
        result = converter.convert_text(text, "test")
        # Using the text comparison utility to handle spacing consistently
        assert_texts_equal(result, "Text with special chars!")

    def test_load_text_file(self, converter, temp_text_file):
        """Test loading and converting a text file."""
        text = converter.load_document(temp_text_file)
        cleaned_text = clean_text_for_comparison(text)
        assert "This is a sample" in cleaned_text
        assert "special characters" in cleaned_text

    def test_unsupported_format(self, converter, tmp_path):
        """Test handling of unsupported file formats."""
        unsupported_file = tmp_path / "test.unsupported"
        unsupported_file.write_text("test")
        with pytest.raises(ValueError, match="Unsupported file format"):
            converter.load_document(unsupported_file)

    def test_missing_file(self, converter):
        """Test handling of missing files."""
        with pytest.raises(FileNotFoundError):
            converter.load_document(Path("nonexistent.txt"))

# DocumentChunker Tests
class TestDocumentChunker:
    def test_chunk_creation_basic(self, chunker):
        """Test basic chunk creation with simple text."""
        text = "Short text for testing chunking functionality."
        chunks = chunker.create_chunks(text, "test.txt")
        assert len(chunks) > 0
        assert_texts_equal(chunks[0].text, text)

    def test_chunk_overlap(self, chunker):
        """Test that chunks properly overlap."""
        # Create text that will definitely create multiple chunks
        long_text = " ".join(["word"] * 200)
        chunks = chunker.create_chunks(long_text, "test.txt")
        assert len(chunks) > 1

        # Check for overlap
        last_words_first_chunk = chunks[0].text.split()[-5:]
        first_words_second_chunk = chunks[1].text.split()[:5]
        assert any(word in first_words_second_chunk for word in last_words_first_chunk)

    def test_chunk_stats(self, chunker):
        """Test chunk statistics calculation."""
        text = "Short text for testing."
        chunks = chunker.create_chunks(text, "test.txt")
        stats = chunker.get_chunk_stats(chunks)

        for key in ['num_chunks', 'avg_length', 'min_length', 'max_length']:
            assert key in stats
        assert stats['num_chunks'] > 0

    def test_empty_text(self, chunker):
        """Test handling of empty text input."""
        chunks = chunker.create_chunks("", "test.txt")
        # Empty text should result in empty chunk list
        assert chunks == []
        # Verify stats can handle empty chunks
        stats = chunker.get_chunk_stats(chunks)
        assert stats == {}

    def test_invalid_chunk_size(self):
        """Test validation of chunk size parameters."""
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=50, chunk_overlap=50)

# DocumentProcessor Tests
class TestDocumentProcessor:
    def test_process_document(self, processor, temp_text_file):
        """Test processing a complete document."""
        chunks = processor.process_document(temp_text_file)
        assert len(chunks) > 0
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)

    def test_process_text(self, processor):
        """Test direct text processing."""
        text = "Sample text for direct processing"
        chunks = processor.process_text(text, "test_source")
        assert len(chunks) > 0
        assert chunks[0].source_doc == "test_source"
        assert text in chunks[0].text
