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
from typing import List, Dict, Any
import re
import torch
import gc

from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.components.converters import TextFileToDocument, PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentPreprocessor:
    """Preprocessor for documents in the FirstRespondersChatbot project."""

    def __init__(
        self, docs_dir: str = "docs", output_dir: str = "data", batch_size: int = 1
    ):
        """
        Initialize the document preprocessor.

        Args:
            docs_dir: Directory containing the documents to preprocess
            output_dir: Directory to save the preprocessed documents
            batch_size: Batch size for processing (defaults to 1 for memory efficiency)
        """
        self.docs_dir = Path(docs_dir)
        self.output_dir = Path(output_dir)
        self.preprocessed_data_path = self.output_dir / "preprocessed_data.json"
        self.batch_size = batch_size

        # Detect if running on Apple Silicon
        self.is_mps = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if self.is_mps:
            logger.info("Apple Silicon detected, using MPS-optimized settings")

        # List of terms specific to first responder domain to preserve during cleaning
        self.domain_terms = [
            "PPE",
            "CPR",
            "AED",
            "EMT",
            "SCBA",
            "EMS",
            "ICS",
            "NFPA",
            "paramedic",
            "firefighter",
            "emergency",
            "trauma",
            "triage",
            "incident command",
            "hazmat",
            "evacuation",
            "extrication",
            # Additional terms optimized for Llama 2 context
            "OSHA",
            "safety protocol",
            "medical emergency",
            "first responder",
            "emergency response",
            "vital signs",
            "emergency medical",
            "rescue operation",
        ]

        # Patterns for identifying book-specific content to clean
        # Enhanced for Llama 2 to improve preprocessing quality
        self.noise_patterns = [
            r"\bpage\s+\d+\b",  # Page numbers
            r"^\s*\d+\s*$",  # Standalone numbers
            r"©.*?reserved",  # Copyright notices
            r"chapter\s+\d+",  # Chapter headers
            r"^\s*figure\s+\d+[-.:]\d+\s*$",  # Figure references
            r"^\s*table\s+\d+[-.:]\d+\s*$",  # Table references
            r"www\.[a-z0-9.-]+\.[a-z]{2,}",  # Websites
            r"^\s*appendix\s+[a-z]\s*$",  # Appendix headers
            r"^\s*section\s+\d+([.]\d+)*\s*$",  # Section numbers
            r"^\s*index\s*$",  # Index header
            r"^\s*references\s*$",  # References header
            r"^\s*bibliography\s*$",  # Bibliography header
        ]

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

        # Get list of files to process
        file_paths = list(self.docs_dir.glob("**/*"))
        file_paths = [f for f in file_paths if f.is_file()]

        # Process files in batches for memory efficiency
        if self.is_mps:
            # Clear MPS memory before starting
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

            # Process in smaller batches on Apple Silicon
            batch_size = 1  # Process one file at a time for M-series chips
        else:
            batch_size = self.batch_size

        # Process files in batches
        for i in range(0, len(file_paths), batch_size):
            batch_files = file_paths[i : i + batch_size]

            # Process each file in the batch
            for file_path in batch_files:
                # Log which file is being processed
                file_count += 1
                logger.info(
                    f"Processing file {file_count}/{len(file_paths)}: {file_path.name}"
                )

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

                    # Optimize the splitter for Llama 2
                    # Use optimized chunk sizes for Apple Silicon
                    if self.is_mps:
                        split_length = 250  # Slightly smaller on MPS for Llama 2
                        split_overlap = 40  # Slightly smaller on MPS
                    else:
                        split_length = 280  # Optimized for Llama 2
                        split_overlap = 45  # Optimized overlap for Llama 2

                    pipeline.add_component(
                        "splitter",
                        DocumentSplitter(
                            split_by="word",
                            split_length=split_length,
                            split_overlap=split_overlap,
                        ),
                    )

                    # Connect components in the pipeline
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

                        # Add metadata and clean documents
                        for i, doc in enumerate(split_documents):
                            if not hasattr(doc, "meta") or doc.meta is None:
                                doc.meta = {}

                            # Extract page number from metadata if available
                            page_num = None
                            if hasattr(doc, "meta") and "page" in doc.meta:
                                page_num = doc.meta["page"]

                            # Enhanced metadata
                            doc.meta["file_name"] = file_path.name
                            doc.meta["file_path"] = str(file_path)
                            doc.meta["chunk_id"] = f"{file_path.stem}_{i}"
                            if page_num:
                                doc.meta["page_number"] = page_num

                            # Apply domain-specific cleaning
                            doc.content = self._clean_content(doc.content)

                        # Filter out very short or low-quality chunks
                        split_documents = [
                            doc
                            for doc in split_documents
                            if len(doc.content.strip()) > 100
                            and self._is_quality_content(doc.content)
                        ]

                        # Log the number of chunks created
                        logger.info(
                            f"Split {file_path.name} into {len(split_documents)} quality chunks"
                        )
                        all_documents.extend(split_documents)
                    else:
                        logger.warning(f"No documents produced for {file_path}")

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    continue

            # For memory management on Apple Silicon
            if self.is_mps:
                # Force garbage collection after each batch on MPS
                gc.collect()
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()

        logger.info(f"Total processed documents: {len(all_documents)}")
        return all_documents

    def _clean_content(self, content: str) -> str:
        """
        Clean document content while preserving domain-specific terminology.
        Optimized for Llama 2 tokenization.

        Args:
            content: The document content to clean

        Returns:
            Cleaned content
        """
        # Preserve domain terms (replace with placeholders)
        placeholders = {}
        for i, term in enumerate(self.domain_terms):
            placeholder = f"__DOMAIN_TERM_{i}__"
            pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            content = pattern.sub(placeholder, content)
            placeholders[placeholder] = term

        # Clean noise patterns
        for pattern in self.noise_patterns:
            content = re.sub(pattern, "", content, flags=re.IGNORECASE | re.MULTILINE)

        # Remove excessive whitespace
        content = re.sub(r"\s+", " ", content).strip()

        # Fix common OCR errors
        content = content.replace("|", "I").replace("l", "I")

        # Remove URL fragments that might have been missed
        content = re.sub(r"http\S+", "", content)

        # Clean up weird punctuation combinations common in OCR
        content = re.sub(r"[.]{2,}", ".", content)  # Multiple periods
        content = re.sub(r"[\s]*[,.][\s]*", ". ", content)  # Weird comma/period spacing

        # Restore domain terms
        for placeholder, term in placeholders.items():
            content = content.replace(placeholder, term)

        return content

    def _is_quality_content(self, content: str) -> bool:
        """
        Check if content is of sufficient quality for training.
        Optimized for Llama 2 training.

        Args:
            content: The document content to check

        Returns:
            True if content is high quality, False otherwise
        """
        # Skip content that's too short
        if len(content.strip()) < 100:
            return False

        # Skip content that's mostly numbers or special characters
        alphanumeric_ratio = (
            sum(c.isalnum() for c in content) / len(content) if content else 0
        )
        if alphanumeric_ratio < 0.5:
            return False

        # Skip content with too many newlines (likely tables or formatting issues)
        newline_ratio = content.count("\n") / len(content) if content else 0
        if newline_ratio > 0.2:
            return False

        # Llama 2 specific: Check for sufficient sentence structure
        # This helps ensure the content is actual text and not just fragments
        sentences = content.split(".")
        if len(sentences) < 2:
            return False

        # Check for at least some meaningful words
        word_count = len(content.split())
        if word_count < 20:
            return False

        return True

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

        # For Apple Silicon, save in smaller batches to manage memory
        if self.is_mps and len(docs_dicts) > 1000:
            # First create temporary batch files
            logger.info(
                f"Saving documents in batches for Apple Silicon memory management"
            )
            batch_size = 1000
            temp_files = []

            for i in range(0, len(docs_dicts), batch_size):
                batch = docs_dicts[i : i + batch_size]
                temp_file = self.output_dir / f"batch_{i//batch_size}.json"
                temp_files.append(temp_file)

                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(batch, f, indent=2)

                # Clear memory after each batch
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()

            # Merge all batches into final file
            merged_docs = []
            for temp_file in temp_files:
                with open(temp_file, "r", encoding="utf-8") as f:
                    batch_docs = json.load(f)
                    merged_docs.extend(batch_docs)
                # Delete temp file
                os.unlink(temp_file)

            # Save merged docs
            with open(self.preprocessed_data_path, "w", encoding="utf-8") as f:
                json.dump(merged_docs, f, indent=2)
        else:
            # Standard save for smaller datasets or non-MPS
            with open(self.preprocessed_data_path, "w", encoding="utf-8") as f:
                json.dump(docs_dicts, f, indent=2)

        logger.info(
            f"Saved {len(docs_dicts)} preprocessed documents to {self.preprocessed_data_path}"
        )

    def run(self) -> None:
        """Run the preprocessing pipeline."""
        logger.info("Starting document preprocessing...")

        # For Apple Silicon, manage memory proactively
        if self.is_mps:
            # Clear MPS memory before starting
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

            # Force garbage collection
            gc.collect()

            logger.info("Memory optimizations enabled for Apple Silicon")

        # Preprocess documents
        processed_documents = self.preprocess_documents()

        # Save documents
        self.save_documents(processed_documents)

        # Final cleanup for Apple Silicon
        if self.is_mps and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
            gc.collect()

        logger.info("Document preprocessing completed successfully!")
