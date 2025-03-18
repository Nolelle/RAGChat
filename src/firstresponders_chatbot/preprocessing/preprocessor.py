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
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner

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
        file_count = 0

        # Process each file in the docs directory
        for file_path in self.docs_dir.glob("**/*"):
            if file_path.is_file():
                # Log which file is being processed
                file_count += 1
                logger.info(f"Processing file {file_count}: {file_path.name}")

                try:
                    # Create a pipeline for each file
                    pipeline = Pipeline()

                    # Add appropriate converter based on file type
                    if file_path.suffix.lower() == ".pdf":
                        pipeline.add_component("converter", PyPDFToDocument())
                    elif file_path.suffix.lower() in [".txt", ".md"]:
                        pipeline.add_component("converter", TextFileToDocument())
                    else:
                        continue

                    # Add a document cleaner
                    pipeline.add_component("cleaner", DocumentCleaner())

                    # Add a single splitter with valid parameters
                    pipeline.add_component(
                        "splitter",
                        DocumentSplitter(
                            split_by="word", split_length=150, split_overlap=30
                        ),
                    )

                    # Try with specific component connections
                    pipeline.connect("converter.documents", "cleaner.documents")
                    pipeline.connect("cleaner.documents", "splitter.documents")

                    # Run pipeline with a single file path
                    logger.debug(f"Running pipeline for {file_path}")
                    result = pipeline.run({"converter": {"sources": [str(file_path)]}})

                    # If documents were successfully processed
                    if "splitter" in result and "documents" in result["splitter"]:
                        split_documents = result["splitter"]["documents"]

                        # Ensure we have a list of documents
                        if not isinstance(split_documents, list):
                            split_documents = [split_documents]

                        # Add metadata to documents
                        for i, doc in enumerate(split_documents):
                            if not hasattr(doc, "meta") or doc.meta is None:
                                doc.meta = {}
                            doc.meta["file_name"] = file_path.name
                            doc.meta["file_path"] = str(file_path)

                        # Log the number of chunks created
                        logger.info(
                            f"Split {file_path.name} into {len(split_documents)} chunks"
                        )
                        all_documents.extend(split_documents)
                    else:
                        logger.warning(f"No documents produced for {file_path}")

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
