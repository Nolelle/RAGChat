#!/usr/bin/env python3
"""
preprocess.py - Preprocessing script for the FirstRespondersChatbot project.

This script preprocesses PDF and text files from the docs/ directory using
Haystack 2.0, converts documents into text, splits them into meaningful chunks,
and saves the output to a JSON file.
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

# File paths
DOCS_DIR = Path("docs")
OUTPUT_DIR = Path("data")
PREPROCESSED_DATA_PATH = OUTPUT_DIR / "preprocessed_data.json"


def preprocess_documents() -> List[Document]:
    """
    Preprocess documents in the docs directory.

    Returns:
        List[Document]: List of processed and split documents.
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_documents = []

    # Process each file in the docs directory
    for file_path in DOCS_DIR.glob("**/*"):
        if file_path.is_file():
            logger.info(f"Processing file: {file_path}")
            try:
                # Create new component instances for each file
                pdf_converter = PyPDFToDocument()
                text_converter = TextFileToDocument()
                splitter = DocumentSplitter(
                    split_by="sentence", split_length=3, split_overlap=1
                )

                # Create a pipeline for each file
                pipeline = Pipeline()
                pipeline.add_component("splitter", splitter)

                if file_path.suffix.lower() == ".pdf":
                    pipeline.add_component("converter", pdf_converter)
                elif file_path.suffix.lower() in [".txt", ".md"]:
                    pipeline.add_component("converter", text_converter)
                else:
                    logger.warning(f"Skipping unsupported file: {file_path}")
                    continue

                # Connect converter to splitter
                pipeline.connect("converter.documents", "splitter.documents")

                # Run pipeline
                result = pipeline.run({"converter": {"sources": [str(file_path)]}})
                split_documents = result["splitter"]["documents"]
                logger.info(
                    f"Split {file_path.name} into {len(split_documents)} chunks"
                )
                all_documents.extend(split_documents)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue

    logger.info(f"Total processed documents: {len(all_documents)}")
    return all_documents


def save_documents(documents: List[Document]) -> None:
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
            "meta": doc.meta,  # Changed from 'metadata' to 'meta'
        }
        docs_dicts.append(doc_dict)

    # Save to JSON file
    with open(PREPROCESSED_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(docs_dicts, f, indent=2)

    logger.info(
        f"Saved {len(docs_dicts)} preprocessed documents to {PREPROCESSED_DATA_PATH}"
    )


def main():
    """Main function to process documents and save them."""
    logger.info("Starting document preprocessing...")

    # Preprocess documents
    processed_documents = preprocess_documents()

    # Save documents
    save_documents(processed_documents)

    logger.info("Document preprocessing completed successfully!")


if __name__ == "__main__":
    main()
