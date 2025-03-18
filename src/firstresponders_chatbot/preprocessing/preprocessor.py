"""
Preprocessing module for the FirstRespondersChatbot project.

This module provides functionality to preprocess PDF and text files using
Haystack 2.0, convert documents into text, split them into meaningful chunks,
and save the output to a JSON file.
"""

import os
import logging
from pathlib import Path
import json
from typing import List

from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.components.converters import TextFileToDocument, PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentPreprocessor:
    """Preprocessor for documents in the FirstRespondersChatbot project."""

    def __init__(self, docs_dir: str = "docs", output_dir: str = "data"):
        """
        Initialize the document preprocessor.

        Args:
            docs_dir: Directory containing the documents to preprocess
            output_dir: Directory to save the preprocessed documents
        """
        self.docs_dir = Path(docs_dir)
        self.output_dir = Path(output_dir)
        self.preprocessed_data_path = self.output_dir / "preprocessed_data.json"

    def preprocess_documents(self) -> List[Document]:
        """
        Preprocess documents in the docs directory.

        Returns:
            List[Document]: List of processed and split documents.
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        all_documents = []

        # Process each file in the docs directory
        for file_path in self.docs_dir.glob("**/*"):
            if file_path.is_file():
                logger.info(f"Processing file: {file_path}")
                try:
                    # Create new component instances for each file
                    pdf_converter = PyPDFToDocument()
                    text_converter = TextFileToDocument()

                    # First split by chapter or section (larger units)
                    section_splitter = DocumentSplitter(
                        split_by="passage",
                        split_length=10,
                        split_overlap=2,
                        add_page_number=True,
                    )

                    # Then split into smaller chunks for more precise retrieval
                    chunk_splitter = DocumentSplitter(
                        split_by="word", split_length=150, split_overlap=30
                    )

                    # Create a pipeline for each file
                    pipeline = Pipeline()

                    # Add appropriate converter based on file type
                    if file_path.suffix.lower() == ".pdf":
                        pipeline.add_component("converter", pdf_converter)
                    elif file_path.suffix.lower() in [".txt", ".md"]:
                        pipeline.add_component("converter", text_converter)
                    else:
                        logger.warning(f"Skipping unsupported file: {file_path}")
                        continue

                    # Add splitters to pipeline
                    pipeline.add_component("section_splitter", section_splitter)
                    pipeline.add_component("chunk_splitter", chunk_splitter)

                    # Connect components
                    pipeline.connect(
                        "converter.documents", "section_splitter.documents"
                    )
                    pipeline.connect(
                        "section_splitter.documents", "chunk_splitter.documents"
                    )

                    # Run pipeline
                    result = pipeline.run({"converter": {"sources": [str(file_path)]}})

                    # Get the final split documents
                    split_documents = result["chunk_splitter"]["documents"]

                    # Add metadata to documents
                    for doc in split_documents:
                        if "meta" not in doc or doc.meta is None:
                            doc.meta = {}
                        doc.meta["file_name"] = file_path.name
                        doc.meta["file_path"] = str(file_path)

                    logger.info(
                        f"Split {file_path.name} into {len(split_documents)} chunks"
                    )
                    all_documents.extend(split_documents)

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    continue

        logger.info(f"Total processed documents: {len(all_documents)}")
        return all_documents

    def save_documents(self, documents: List[Document]) -> None:
        """
        Save processed documents to a JSON file.

        Args:
            documents: List of processed documents.
        """
        # Convert documents to dictionaries for JSON serialization
        docs_dicts = []
        for i, doc in enumerate(documents):
            doc_dict = {
                "id": doc.id if doc.id else f"doc_{i}",
                "content": doc.content,
                "meta": doc.meta,
            }
            docs_dicts.append(doc_dict)

        # Save to JSON file
        with open(self.preprocessed_data_path, "w", encoding="utf-8") as f:
            json.dump(docs_dicts, f, indent=2)

        logger.info(
            f"Saved {len(docs_dicts)} preprocessed documents to {self.preprocessed_data_path}"
        )

    def run(self) -> None:
        """Run the preprocessing pipeline."""
        logger.info("Starting document preprocessing...")

        # Preprocess documents
        processed_documents = self.preprocess_documents()

        # Save documents
        self.save_documents(processed_documents)

        logger.info("Document preprocessing completed successfully!")
