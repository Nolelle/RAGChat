import os
import logging
from pathlib import Path
from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def detect_garbled_text(text: str) -> bool:
    """
    Detect if text is likely garbled due to encoding issues.
    Uses entropy and character distribution analysis.

    Args:
        text: The text to check

    Returns:
        bool: True if the text appears to be garbled
    """
    if not text or len(text) < 20:
        return False

    # Check for high percentage of non-ASCII characters (potential encoding issues)
    non_ascii_count = sum(1 for c in text if ord(c) > 127)
    if len(text) > 0 and non_ascii_count / len(text) > 0.3:  # If > 30% non-ASCII
        return True

    # Check for unusual character distribution (Shannon entropy)
    import math
    from collections import Counter

    # Calculate character frequency
    char_counts = Counter(text)
    total_chars = len(text)

    # Calculate entropy
    entropy = 0
    for count in char_counts.values():
        prob = count / total_chars
        entropy -= prob * math.log2(prob)

    # English text typically has entropy between 3.5-5.0
    # Garbled text or encrypted content often has higher entropy (>5.5)
    if entropy > 5.5:
        return True

    # Check for unusual character sequences (random distribution of special chars)
    import re

    # Look for sequences like "iaIAs" or "síaà" that are unlikely in normal text
    unusual_pattern = r"([^\w\s]{3,}|([a-zA-Z][^a-zA-Z]){4,})"
    if re.search(unusual_pattern, text):
        return True

    return False


def process_pdf_file(file_path: str):
    """Test processing a PDF file with our improved method."""
    logger.info(f"Testing PDF processing on: {file_path}")

    try:
        # Create pipeline components
        pdf_converter = PyPDFToDocument()

        # Create a cleaner component for PDFs to handle special characters and formatting
        cleaner = DocumentCleaner(
            remove_empty_lines=True,
            remove_extra_whitespaces=True,
        )

        # Use a conservative splitting approach for PDFs
        splitter = DocumentSplitter(
            split_by="passage",  # Split by passage for better context preservation
            split_length=200,  # Larger chunks to maintain context
            split_overlap=50,  # Increased overlap to prevent context loss
        )

        # Create a pipeline
        pipeline = Pipeline()

        # Add components
        pipeline.add_component("converter", pdf_converter)
        pipeline.add_component("cleaner", cleaner)
        pipeline.add_component("splitter", splitter)

        # Connect components in the pipeline
        pipeline.connect("converter.documents", "cleaner.documents")
        pipeline.connect("cleaner.documents", "splitter.documents")

        # Run pipeline
        logger.info("Starting document conversion pipeline...")
        result = pipeline.run({"converter": {"sources": [file_path]}})

        # Get the documents
        if "splitter" in result and "documents" in result["splitter"]:
            documents = result["splitter"]["documents"]

            # Log counts
            logger.info(f"Number of chunks: {len(documents)}")

            # Process and clean each document
            cleaned_docs = []
            garbled_docs = 0

            for i, doc in enumerate(documents):
                content = doc.content

                # Check if content appears garbled
                is_garbled = detect_garbled_text(content)
                if is_garbled:
                    garbled_docs += 1
                    logger.warning(f"Document {i+1} contains garbled text")

                # Clean content
                import re

                content = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", content)
                content = content.replace("\ufeff", "")  # Zero-width no-break space

                # Handle common encoding issues
                content = content.replace("â€™", "'")  # Smart single quote
                content = content.replace("â€œ", '"')  # Smart left double quote
                content = content.replace("â€", '"')  # Smart right double quote
                content = content.replace('â€"', "–")  # En dash
                content = content.replace('â€"', "—")  # Em dash

                # Replace various other problematic characters
                content = content.replace("Ã©", "é")
                content = content.replace("Ã¨", "è")
                content = content.replace("Ã", "à")
                content = content.replace("Ã¢", "â")
                content = content.replace("Ã®", "î")
                content = content.replace("Ã´", "ô")
                content = content.replace("Ã»", "û")
                content = content.replace("Ã§", "ç")

                # Replace consecutive whitespace with single space
                content = re.sub(r"\s+", " ", content)

                # Check if content is still garbled after cleaning
                is_still_garbled = detect_garbled_text(content)
                if is_still_garbled:
                    logger.warning(
                        f"Document {i+1} still contains garbled text after cleaning"
                    )
                    # Apply more aggressive cleaning
                    content = re.sub(r"[^\x20-\x7E]", "", content)

                # Print samples
                if i < 5:  # Print first 5 docs
                    logger.info(f"Document {i+1} sample content: {content[:200]}...")

                cleaned_docs.append(content)

            # Log stats
            logger.info(f"Processed {len(documents)} documents")
            logger.info(f"Found {garbled_docs} documents with garbled text")

            # Check for SCBA content
            scba_docs = []
            doffing_docs = []

            for i, content in enumerate(cleaned_docs):
                # Check for any SCBA content
                if (
                    "SCBA" in content
                    or "self-contained breathing apparatus" in content.lower()
                ):
                    logger.info(f"Found SCBA content in document {i+1}")
                    scba_docs.append(content)

                    # Check specifically for doffing procedure
                    if (
                        "doff" in content.lower() or "remov" in content.lower()
                    ) and "SCBA" in content:
                        logger.info(f"Found SCBA DOFFING content in document {i+1}")
                        doffing_docs.append(content)

            logger.info(f"Found {len(scba_docs)} documents containing SCBA content")
            logger.info(
                f"Found {len(doffing_docs)} documents containing SCBA doffing procedures"
            )

            # If SCBA docs found, print samples
            for i, content in enumerate(scba_docs[:3]):
                logger.info(f"SCBA Document {i+1} content: {content[:300]}")

            # If doffing docs found, print samples
            for i, content in enumerate(doffing_docs[:3]):
                logger.info(f"SCBA DOFFING Document {i+1} content: {content[:500]}")

            return len(scba_docs) > 0

        else:
            logger.error("No documents were produced by the pipeline")
            return False

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def main():
    """Run the test."""
    # Get all PDF files in uploads directory
    uploads_dir = Path("uploads")
    pdf_files = list(uploads_dir.glob("*.pdf"))

    if not pdf_files:
        logger.error("No PDF files found in uploads directory")
        return

    # Process each PDF file
    for pdf_file in pdf_files:
        logger.info(f"Processing file: {pdf_file}")
        success = process_pdf_file(str(pdf_file))
        if success:
            logger.info(f"Successfully processed file: {pdf_file}")
        else:
            logger.error(f"Failed to process file: {pdf_file}")


if __name__ == "__main__":
    main()
