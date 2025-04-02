"""
RAG system module for the FirstRespondersChatbot project.

This module implements the backend logic for the Retrieval-Augmented Generation (RAG)
system, indexing uploaded PDF/text files, retrieving relevant context, and generating
responses using the fine-tuned Llama 2 model.
"""

import logging
import os
import math
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import uuid
from collections import defaultdict
import re
import time

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

from haystack import Pipeline, Document
from haystack.components.converters import TextFileToDocument, PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers import (
    InMemoryEmbeddingRetriever,
    InMemoryBM25Retriever,
)
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG system for the FirstRespondersChatbot."""

    def __init__(
        self,
        model_name_or_path: str = "trained-models/phi4-mini-first-responder",
        device: str = None,
        document_store_dir: str = "uploads",
        use_mps: bool = True,
        top_k: int = 5,
        rerank_top_k: int = 5,
    ):
        """
        Initialize the RAG system.

        Args:
            model_name_or_path: Path to the fine-tuned model or HF model name (e.g., 'trained-models' or 'microsoft/Phi-4-mini-instruct')
            device: Device to use for model inference
            document_store_dir: Directory for the document store
            use_mps: Whether to use MPS (Apple Silicon) when available
            top_k: Number of documents to retrieve
            rerank_top_k: Number of documents to return after reranking
        """
        self.model_name = model_name_or_path
        self.document_store_dir = Path(document_store_dir)
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.indexed_files = set()  # Keep track of indexed files
        self.session_files = defaultdict(set)  # Files by session

        # Create uploads directory if it doesn't exist
        os.makedirs(self.document_store_dir, exist_ok=True)

        # Set up device
        self.device = self._setup_device(use_mps, device)

        # Initialize tokenizer and model
        self._init_model_and_tokenizer()

        # Initialize RAG components
        self._init_rag_components()

    def _setup_device(self, use_mps: bool, device: str = None) -> torch.device:
        """Set up the device for model inference."""
        if device:
            return torch.device(device)
        elif use_mps and torch.backends.mps.is_available():
            logger.info("Using Apple Silicon acceleration")
            return torch.device("mps")
        elif torch.cuda.is_available():
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        else:
            logger.info("Using CPU")
            return torch.device("cpu")

    def _init_model_and_tokenizer(self):
        """Initialize the Phi-4 mini model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")

            # Configure model loading
            torch_dtype = (
                torch.float16 if self.device.type in ["cuda", "mps"] else torch.float32
            )

            # Configure quantization only if NOT on MPS
            quantization_config = None  # Default to no quantization config
            if self.device.type != "mps":
                logger.info("Using 4-bit quantization for model loading")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                logger.info(
                    "Skipping bitsandbytes quantization for MPS, using float16 directly"
                )

            # Load model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,  # Will be None for MPS
                trust_remote_code=True,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # Ensure pad token is set (important for Llama 2)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Log the chat template associated with the tokenizer
            logger.info(f"Tokenizer chat template: {self.tokenizer.chat_template}")

            logger.info(f"Successfully loaded model and tokenizer")

        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _init_rag_components(self):
        """Initialize RAG components."""
        # Initialize document store
        self.document_store = InMemoryDocumentStore()

        # Initialize embedders
        self.document_embedder = SentenceTransformersDocumentEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_embedder = SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize retrievers
        self.embedding_retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store
        )
        self.bm25_retriever = InMemoryBM25Retriever(document_store=self.document_store)

        # Initialize reranker
        self.reranker = TransformersSimilarityRanker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=self.rerank_top_k
        )

        # Warm up the embedding models
        self.warm_up()

        logger.info("RAG system initialized successfully")

    def warm_up(self):
        """
        Warm up the embedding models by calling their warm_up() methods.
        This ensures the models are loaded before they are used.
        """
        try:
            logger.info("Warming up embedding models and reranker...")
            # Warm up document embedder
            self.document_embedder.warm_up()

            # Warm up text embedder
            self.text_embedder.warm_up()

            # Warm up reranker
            self.reranker.warm_up()

            logger.info("Embedding models and reranker warmed up successfully")
        except Exception as e:
            logger.error(f"Error warming up models: {str(e)}")
            raise

    def _detect_garbled_text(self, text: str, is_model_output: bool = False) -> bool:
        """
        Detect if text is likely garbled due to encoding issues.
        Uses entropy and character distribution analysis.

        Args:
            text: The text to check
            is_model_output: Whether this text is from the model (less stringent checks)

        Returns:
            bool: True if the text appears to be garbled
        """
        if not text or len(text) < 20:
            return False

        # For debugging: Log what we're checking
        debug_prefix = "MODEL OUTPUT" if is_model_output else "DOCUMENT TEXT"
        logger.debug(f"DEBUG: {debug_prefix} GARBLED CHECK - Length: {len(text)}")
        logger.debug(f"DEBUG: {debug_prefix} GARBLED CHECK - Preview: {text[:100]}...")

        # For model output, do a quick check for reasonable text
        if is_model_output:
            # If it contains common words and reasonable sentence structure, it's likely fine
            common_words = [
                "the",
                "and",
                "for",
                "this",
                "that",
                "with",
                "you",
                "are",
                "is",
                "not",
                "have",
            ]
            common_words_count = sum(
                1 for word in common_words if f" {word} " in f" {text} ".lower()
            )

            logger.debug(
                f"DEBUG: {debug_prefix} GARBLED CHECK - Common words found: {common_words_count}"
            )

            # If it has several common words, it's probably not garbled
            if common_words_count >= 3:
                # Also check for proper punctuation and sentence structure
                sentences = re.split(r"[.!?]+", text)
                proper_sentences = sum(
                    1
                    for s in sentences
                    if len(s.strip()) > 10 and s.strip()[0].isupper()
                )

                logger.debug(
                    f"DEBUG: {debug_prefix} GARBLED CHECK - Proper sentences found: {proper_sentences}/{len(sentences)}"
                )

                # If we have proper sentences, consider it valid
                if proper_sentences >= 1:
                    logger.debug(
                        f"DEBUG: {debug_prefix} GARBLED CHECK - Text passes common word and sentence structure checks"
                    )
                    return False

        # Check for high percentage of non-ASCII characters (potential encoding issues)
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        non_ascii_percent = non_ascii_count / len(text) if len(text) > 0 else 0
        ascii_threshold = (
            0.3 if not is_model_output else 0.4
        )  # Less strict for model output

        logger.debug(
            f"DEBUG: {debug_prefix} GARBLED CHECK - Non-ASCII chars: {non_ascii_count}/{len(text)} ({non_ascii_percent*100:.2f}%)"
        )
        logger.debug(
            f"DEBUG: {debug_prefix} GARBLED CHECK - ASCII threshold: {ascii_threshold*100:.2f}%"
        )

        if len(text) > 0 and non_ascii_percent > ascii_threshold:
            logger.debug(
                f"DEBUG: {debug_prefix} GARBLED CHECK - Failed ASCII check: Too many non-ASCII characters"
            )
            return True

        # Check for unusual character distribution (Shannon entropy)
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
        # Use different thresholds based on source
        entropy_threshold = (
            5.5 if not is_model_output else 6.0
        )  # Higher threshold for model output

        logger.debug(
            f"DEBUG: {debug_prefix} GARBLED CHECK - Text entropy: {entropy:.2f}"
        )
        logger.debug(
            f"DEBUG: {debug_prefix} GARBLED CHECK - Entropy threshold: {entropy_threshold:.2f}"
        )

        if entropy > entropy_threshold:
            logger.debug(
                f"DEBUG: {debug_prefix} GARBLED CHECK - Failed entropy check: Entropy too high"
            )
            return True

        # Check for unusual character sequences (random distribution of special chars)
        # Look for sequences like "iaIAs" or "síaà" that are unlikely in normal text
        unusual_pattern = r"([^\w\s]{3,}|([a-zA-Z][^a-zA-Z]){4,})"  # Reduced sequence length requirements
        unusual_sequences = re.findall(unusual_pattern, text)

        logger.debug(
            f"DEBUG: {debug_prefix} GARBLED CHECK - Unusual sequences found: {len(unusual_sequences)}"
        )
        if unusual_sequences:
            logger.debug(
                f"DEBUG: {debug_prefix} GARBLED CHECK - Examples of unusual sequences: {unusual_sequences[:3]}"
            )

        if re.search(unusual_pattern, text):
            logger.debug(
                f"DEBUG: {debug_prefix} GARBLED CHECK - Failed pattern check: Unusual character sequences found"
            )
            return True

        # Additional check for concatenated words without spaces (common in bad PDF extraction)
        words_without_spaces = re.findall(r"[a-zA-Z]{20,}", text)
        words_without_spaces_percent = (
            len("".join(words_without_spaces)) / len(text) if len(text) > 0 else 0
        )

        logger.debug(
            f"DEBUG: {debug_prefix} GARBLED CHECK - Long words without spaces: {len(words_without_spaces)}"
        )
        if words_without_spaces:
            logger.debug(
                f"DEBUG: {debug_prefix} GARBLED CHECK - Examples of words without spaces: {words_without_spaces[:3]}"
            )
        logger.debug(
            f"DEBUG: {debug_prefix} GARBLED CHECK - Words without spaces percentage: {words_without_spaces_percent*100:.2f}%"
        )

        if words_without_spaces and words_without_spaces_percent > 0.2:
            logger.debug(
                f"DEBUG: {debug_prefix} GARBLED CHECK - Failed spaces check: Too many concatenated words"
            )
            return True

        logger.debug(f"DEBUG: {debug_prefix} GARBLED CHECK - Text passed all checks")
        return False

    def process_file(self, file_path: str) -> List[Document]:
        """
        Process a file and split it into documents.

        Args:
            file_path: Path to the file to process

        Returns:
            List[Document]: List of processed documents
        """
        logger.info(f"[process_file] Starting processing for file: {file_path}")

        try:
            # Create pipeline components
            pdf_converter = PyPDFToDocument()
            text_converter = TextFileToDocument()

            # Use different splitting parameters for PDFs vs text files
            file_path_obj = Path(file_path)
            if file_path_obj.suffix.lower() == ".pdf":
                logger.info(
                    "[process_file] PDF file detected - using PyPDFToDocument converter with cleaning"
                )

                # Create a cleaner component for PDFs to handle special characters and formatting
                cleaner = DocumentCleaner(
                    remove_empty_lines=True,
                    remove_extra_whitespaces=True,
                )

                # Use a more conservative splitting approach for PDFs
                splitter = DocumentSplitter(
                    split_by="passage",  # Split by passage for better context preservation
                    split_length=400,  # Larger chunks to maintain context (increased from 200)
                    split_overlap=100,  # Increased overlap to prevent context loss (increased from 50)
                )
            else:
                # Use same passage-based splitting for text files but with slightly smaller chunks
                logger.info(
                    "Text file detected - using standardized splitting parameters"
                )
                splitter = DocumentSplitter(
                    split_by="passage",  # Changed from "sentence" to "passage" for better context
                    split_length=250,  # Increased from 150 for better context preservation
                    split_overlap=50,  # Increased from 30 for better overlap
                )

            # Create a pipeline
            pipeline = Pipeline()

            # Add appropriate converter based on file type
            if file_path_obj.suffix.lower() == ".pdf":
                logger.info(
                    "[process_file] PDF file detected - using PyPDFToDocument converter with cleaning"
                )

                # For PDFs, perform pre-check with PyPDF2 to assess potential issues
                try:
                    import PyPDF2

                    with open(file_path, "rb") as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        num_pages = len(pdf_reader.pages)
                        logger.info(f"[process_file] PDF has {num_pages} pages")

                        # Try to extract text from first page to check for encoding issues
                        first_page_text = pdf_reader.pages[0].extract_text()
                        if self._detect_garbled_text(first_page_text):
                            logger.warning(
                                "[process_file] PDF first page contains potential encoding issues"
                            )
                        else:
                            logger.info(
                                "[process_file] PDF first page text seems okay."
                            )
                except Exception as e:
                    logger.warning(
                        f"[process_file] Could not perform PDF pre-check: {str(e)}"
                    )

                pipeline.add_component("converter", pdf_converter)
                pipeline.add_component("cleaner", cleaner)
                pipeline.add_component("splitter", splitter)

                # Connect components in the pipeline
                pipeline.connect("converter.documents", "cleaner.documents")
                pipeline.connect("cleaner.documents", "splitter.documents")

                # Log PDF-specific details
                logger.info(
                    f"[process_file] PDF file size: {os.path.getsize(file_path)} bytes"
                )
            elif file_path_obj.suffix.lower() in [".txt", ".md"]:
                pipeline.add_component("converter", text_converter)
                pipeline.add_component("splitter", splitter)

                # Connect components in the pipeline
                pipeline.connect("converter.documents", "splitter.documents")
            else:
                logger.warning(f"Unsupported file type: {file_path}")
                return []

            # Run pipeline
            logger.info("[process_file] Starting document conversion pipeline...")
            result = pipeline.run({"converter": {"sources": [file_path]}})
            logger.info(
                f"[process_file] Pipeline run completed. Result keys: {result.keys()}"
            )

            if "converter" in result:
                converter_output = result["converter"].get("documents", [])
                logger.info(
                    f"[process_file] Converter output: {len(converter_output)} documents"
                )
                # Log sample of converted content
                if converter_output:
                    sample = (
                        converter_output[0].content[:500]
                        if converter_output[0].content
                        else "<empty content>"
                    )
                    logger.info(f"[process_file] Sample converted content: {sample}...")

                    # Early detection for severely corrupted PDFs
                    if (
                        file_path_obj.suffix.lower() == ".pdf"
                        and self._detect_garbled_text(sample)
                    ):
                        logger.warning(
                            "[process_file] Initial PDF conversion appears to contain garbled text"
                        )
                else:
                    logger.warning(
                        "[process_file] Converter did not produce any documents."
                    )
            else:
                logger.warning(
                    "[process_file] 'converter' key not found in pipeline result."
                )

            # Check if splitter produced documents
            if "splitter" not in result or not result["splitter"].get("documents"):
                logger.warning(
                    "[process_file] Splitter did not produce documents or 'splitter' key missing."
                )
                if file_path_obj.suffix.lower() == ".pdf":
                    logger.error(
                        "[process_file] Splitter did not produce any documents. Attempting fallback PDF extraction using PyPDF2."
                    )
                    try:
                        import PyPDF2

                        with open(file_path, "rb") as f:
                            pdf_reader = PyPDF2.PdfReader(f)
                            all_text = ""
                            for page in pdf_reader.pages:
                                page_text = page.extract_text() or ""
                                all_text += page_text + "\n"
                        if len(all_text.strip()) < 20:
                            logger.error(
                                "[process_file] Fallback extraction produced insufficient text."
                            )
                            return []
                        paragraphs = [
                            p.strip() for p in all_text.split("\n\n") if p.strip()
                        ]
                        fallback_docs = [
                            Document(content=para, meta={"fallback": True})
                            for para in paragraphs
                            if len(para) > 20
                        ]
                        logger.info(
                            f"[process_file] Fallback extraction produced {len(fallback_docs)} documents."
                        )
                        result["splitter"] = {"documents": fallback_docs}
                    except Exception as e:
                        logger.error(
                            "[process_file] Fallback extraction failed: " + str(e)
                        )
                        return []
                else:
                    logger.error(
                        "[process_file] Splitter did not produce any documents for non-PDF file."
                    )
                    return []  # Return empty list if splitter fails for non-PDF

            # Ensure documents are correctly retrieved, even after fallback
            documents = result.get("splitter", {}).get("documents", [])
            if not documents:
                logger.error(
                    "[process_file] No documents found after splitter/fallback stage."
                )
                return []

            # Log chunk details
            logger.info(f"[process_file] Document splitting results:")
            logger.info(
                f"[process_file] Number of chunks before cleaning: {len(documents)}"
            )
            if documents:
                avg_chunk_size = (
                    sum(len(doc.content) for doc in documents if doc.content)
                    / len(documents)
                    if len(documents) > 0
                    else 0
                )
                logger.info(
                    f"[process_file] Average chunk size: {avg_chunk_size:.2f} characters"
                )
                logger.info(
                    f"[process_file] Sample chunk (pre-cleaning): {documents[0].content[:200] if documents[0].content else '<empty>'}..."
                )
            else:  # Added else block
                logger.warning(
                    "[process_file] No documents to calculate chunk details from."
                )

            # Clean up each document's content to handle special characters and encoding issues
            cleaned_documents = []
            garbled_chunks_count = 0

            for doc in documents:
                # Get the content and perform additional cleaning
                content = doc.content

                # Check if content appears garbled
                is_garbled = self._detect_garbled_text(content)
                if is_garbled:
                    garbled_chunks_count += 1
                    logger.warning(
                        f"[process_file] Detected likely garbled text in document chunk. ID: {doc.id}"
                    )

                # Clean content - enhanced cleaning procedure
                # First, clean control characters
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

                # Additional common replacements for PDF encoding issues
                content = content.replace("ï¬", "fi")  # fi ligature
                content = content.replace("ï¬€", "ff")  # ff ligature
                content = content.replace("ï¬‚", "fl")  # fl ligature
                content = content.replace("Ì¶", "")  # Combining macron below
                content = content.replace("Ì©", "")  # Combining vertical line below

                # More aggressive cleaning for concatenated words - add spaces between lowercase-uppercase transitions
                content = re.sub(r"([a-z])([A-Z])", r"\1 \2", content)

                # Replace consecutive whitespace with single space
                content = re.sub(r"\s+", " ", content)

                # Check if content is still garbled after cleaning
                is_still_garbled = self._detect_garbled_text(content)

                # If content is still garbled, try more aggressive cleaning or skip
                if is_still_garbled:
                    garbled_chunks_count += 1

                    # Try more aggressive cleaning for heavily corrupted text
                    # Only keep ASCII characters, digits, and basic punctuation
                    content = re.sub(r"[^\x20-\x7E]", "", content)

                    # Try to fix common patterns in garbled PDF text
                    # Break concatenated words (common in poorly encoded PDFs)
                    content = re.sub(r"([a-z]{2,})([A-Z])", r"\1 \2", content)

                    # Check again after aggressive cleaning
                    if self._detect_garbled_text(content) or len(content.strip()) < 20:
                        # Skip this chunk if it's still garbled or too short after cleaning
                        logger.warning(
                            f"[process_file] Skipping garbled chunk that couldn't be cleaned. ID: {doc.id}"
                        )
                        continue  # Continue to next document

                # Create a new document with the cleaned content
                cleaned_doc = Document(content=content, meta=doc.meta)
                cleaned_doc.id = doc.id

                # Add to cleaned documents list if content is not empty
                if cleaned_doc.content and len(cleaned_doc.content.strip()) > 0:
                    cleaned_documents.append(cleaned_doc)
                else:
                    logger.warning(
                        f"[process_file] Skipping document with empty content after cleaning. ID: {doc.id}"
                    )

            logger.info(
                f"[process_file] Total garbled chunks detected: {garbled_chunks_count}"
            )
            logger.info(
                f"[process_file] Number of documents after cleaning: {len(cleaned_documents)}"
            )

            # Check if we have at least some usable documents
            if not cleaned_documents:
                logger.warning(
                    "[process_file] No usable documents remaining after cleaning process."
                )
                if file_path_obj.suffix.lower() == ".pdf":
                    logger.warning(
                        "[process_file] Cleaning removed all document content. Attempting fallback extraction using PyPDF2 for the PDF (again)."  # Added context
                    )
                    try:
                        import PyPDF2

                        with open(file_path, "rb") as f:
                            pdf_reader = PyPDF2.PdfReader(f)
                            all_text = ""
                            for page in pdf_reader.pages:
                                page_text = page.extract_text() or ""
                                all_text += page_text + "\n"
                        if len(all_text.strip()) < 20:
                            logger.error(
                                "[process_file] Fallback extraction produced insufficient text."
                            )
                            return []
                        paragraphs = [
                            p.strip() for p in all_text.split("\n\n") if p.strip()
                        ]
                        fallback_docs = [
                            Document(content=para, meta={"fallback": True})
                            for para in paragraphs
                            if len(para) > 20
                        ]
                        if not fallback_docs:
                            logger.error(
                                "[process_file] Fallback extraction after cleaning produced no documents."
                            )
                            return []
                        cleaned_documents = fallback_docs
                        logger.info(
                            f"[process_file] Fallback extraction after cleaning produced {len(cleaned_documents)} documents."
                        )
                    except Exception as e:
                        logger.error(
                            "[process_file] Fallback extraction after cleaning failed: "
                            + str(e)
                        )
                        return []
                else:
                    # If not a PDF and no documents, return empty
                    logger.error(
                        "[process_file] No documents after cleaning for non-PDF file."
                    )
                    return []

            logger.info(
                f"[process_file] Returning {len(cleaned_documents)} processed documents for {file_path}"
            )
            return cleaned_documents

        except Exception as e:
            logger.error(
                f"[process_file] Unexpected error processing file {file_path}: {str(e)}"
            )
            logger.exception(
                "[process_file] Full traceback for process_file error:"
            )  # Log full traceback
            return []

    def index_file(self, file_path: str, session_id: str = "default") -> bool:
        """
        Index a file for retrieval.

        Args:
            file_path: Path to the file to index
            session_id: Unique session ID to track files per session

        Returns:
            bool: True if indexing was successful, False otherwise
        """
        logger.info(
            f"[index_file] Starting indexing for file: {file_path}, session: {session_id}"
        )
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"[index_file] File not found: {file_path}")
                return False

            # Initialize session if not exists
            if session_id not in self.session_files:
                logger.info(f"[index_file] Initializing new session: {session_id}")
                self.session_files[session_id] = set()

            # For PDFs, perform a pre-check to verify readability
            file_path_obj = Path(file_path)
            if file_path_obj.suffix.lower() == ".pdf":
                logger.info(f"[index_file] Performing PDF pre-check for: {file_path}")
                try:
                    import PyPDF2

                    with open(file_path, "rb") as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        num_pages = len(pdf_reader.pages)
                        logger.info(
                            f"[index_file] PDF pre-check: {file_path} has {num_pages} pages"
                        )

                        # Attempt to extract text from the first page
                        if num_pages > 0:  # Check if there are pages
                            first_page_text = pdf_reader.pages[0].extract_text()
                            if not first_page_text or len(first_page_text.strip()) < 10:
                                logger.warning(
                                    f"[index_file] PDF pre-check: {file_path} contains very little text in first page"
                                )
                            else:
                                logger.info(
                                    "[index_file] PDF pre-check: First page text seems okay."
                                )
                        else:
                            logger.warning(
                                f"[index_file] PDF pre-check: {file_path} has 0 pages."
                            )
                except Exception as e:
                    logger.warning(
                        f"[index_file] PDF pre-check failed for {file_path}: {str(e)}"
                    )

            # Process file
            logger.info(f"[index_file] Calling process_file for: {file_path}")
            documents = self.process_file(file_path)
            if not documents:
                logger.error(
                    f"[index_file] process_file returned no documents for {file_path}"
                )
                return False

            logger.info(
                f"[index_file] Received {len(documents)} documents from process_file"
            )
            if documents:  # Check if documents list is not empty
                logger.info(
                    f"[index_file] Sample document content after process_file: {documents[0].content[:200] if documents[0].content else '<empty>'}..."
                )

            # Double-check document quality
            valid_docs = 0
            for idx, doc in enumerate(documents):  # Added enumerate for index logging
                is_valid = (
                    doc.content
                    and len(doc.content.strip()) > 20
                    and not self._detect_garbled_text(doc.content)
                )
                if is_valid:
                    valid_docs += 1
                else:
                    logger.warning(
                        f"[index_file] Document {idx} failed quality check. Content sample: {doc.content[:100] if doc.content else '<empty>'}..."
                    )

            logger.info(
                f"[index_file] Quality check: {valid_docs} out of {len(documents)} documents are valid."
            )
            quality_threshold = 0.3
            if valid_docs < len(documents) * quality_threshold:  # Less than 30% valid
                logger.warning(
                    f"[index_file] Only {valid_docs} out of {len(documents)} documents appear to be valid quality (threshold: {quality_threshold*100}%)"
                )
                if valid_docs == 0:
                    logger.error(
                        "[index_file] No valid documents found after quality check, aborting indexing"
                    )
                    return False
                else:  # Allow indexing if some docs are valid, but log a warning
                    logger.warning(
                        "[index_file] Proceeding with indexing despite low quality document ratio."
                    )

            # Add session ID to document metadata
            logger.info(f"[index_file] Adding metadata to {len(documents)} documents.")
            for doc in documents:
                doc.meta["session_id"] = session_id
                doc.meta["file_name"] = os.path.basename(file_path)
                doc.meta["file_path"] = str(file_path)

            # Log document details before embedding
            logger.info("[index_file] Document details before embedding (sample):")
            for i, doc in enumerate(documents[:3]):
                logger.info(
                    f"[index_file] Document {i+1} content preview: {doc.content[:100] if doc.content else '<empty>'}..."
                )
                logger.info(f"[index_file] Document {i+1} metadata: {doc.meta}")

            # Embed documents
            logger.info(
                f"[index_file] Starting document embedding for {len(documents)} documents..."
            )
            embedding_result = self.document_embedder.run(documents=documents)
            if "documents" not in embedding_result:
                logger.error(
                    "[index_file] Embedding step did not return 'documents' key."
                )
                return False

            embedded_documents = embedding_result["documents"]
            logger.info(
                f"[index_file] Embedding completed. Received {len(embedded_documents)} embedded documents."
            )

            # Verify embeddings
            missing_embeddings = 0
            for i, doc in enumerate(embedded_documents):
                if not hasattr(doc, "embedding") or doc.embedding is None:
                    missing_embeddings += 1
                    logger.error(
                        f"[index_file] Document {i} (ID: {doc.id}) is missing embeddings!"
                    )
                elif i < 3:  # Log first 3 docs
                    embedding_shape = (
                        len(doc.embedding) if isinstance(doc.embedding, list) else "N/A"
                    )
                    embedding_sample = (
                        str(doc.embedding[:5])
                        if isinstance(doc.embedding, list)
                        else "N/A"
                    )
                    logger.info(
                        f"[index_file] Document {i} embedding shape: {embedding_shape}"
                    )
                    logger.info(
                        f"[index_file] Document {i} embedding sample: {embedding_sample}..."
                    )

            if missing_embeddings > 0:
                logger.error(
                    f"[index_file] {missing_embeddings} documents are missing embeddings! Aborting indexing."
                )
                return False
            elif len(embedded_documents) != len(documents):
                logger.warning(
                    f"[index_file] Number of embedded documents ({len(embedded_documents)}) does not match original documents ({len(documents)}). Proceeding cautiously."
                )

            # Write to document store
            initial_count = self.document_store.count_documents()
            logger.info(
                f"[index_file] Document store count before writing: {initial_count}"
            )

            # Write new documents
            logger.info(
                f"[index_file] Writing {len(embedded_documents)} documents to the store..."
            )
            write_result = self.document_store.write_documents(embedded_documents)
            logger.info(
                f"[index_file] Document store write_documents result: {write_result}"
            )  # Log write result if available

            final_count = self.document_store.count_documents()
            logger.info(
                f"[index_file] Document store count after writing: {final_count}"
            )

            documents_added = final_count - initial_count
            logger.info(f"[index_file] Added {documents_added} new documents to store.")

            # Verify documents were added
            if (
                documents_added <= 0 and len(embedded_documents) > 0
            ):  # Check if we expected to add docs
                logger.error(
                    "[index_file] No new documents were added to the store despite having processed documents."
                )
                # Optional: Add check for existing documents if overwriting is intended
                # existing_docs = self.document_store.filter_documents(filters={"field": "file_path", "operator": "==", "value": str(file_path)})
                # if len(existing_docs) == len(embedded_documents):
                #    logger.info("[index_file] Documents might have been overwritten.")
                # else:
                #    return False # Still consider it a failure if counts don't match expected state
                return False  # Treat as failure for now

            # Mark file as indexed for this session
            self.indexed_files.add(
                file_path
            )  # This tracks globally, might need refinement if sessions are strictly isolated
            self.session_files[session_id].add(file_path)
            logger.info(
                f"[index_file] Marked '{os.path.basename(file_path)}' as indexed for session '{session_id}'. Total files in session: {len(self.session_files[session_id])}"
            )

            # Verify indexed documents for this session and file
            logger.info(
                f"[index_file] Verifying documents in store for file: {file_path}, session: {session_id}"
            )
            session_file_docs = 0
            try:
                # Use filter_documents which might be more reliable depending on store implementation
                filtered_docs = self.document_store.filter_documents(
                    filters={
                        "operator": "AND",
                        "conditions": [
                            {
                                "field": "meta.session_id",
                                "operator": "==",
                                "value": session_id,
                            },
                            {
                                "field": "meta.file_path",
                                "operator": "==",
                                "value": str(file_path),
                            },
                        ],
                    }
                )
                session_file_docs = len(filtered_docs)
                # Log sample IDs of verified docs
                if session_file_docs > 0:
                    sample_ids = [d.id for d in filtered_docs[:3]]
                    logger.info(
                        f"[index_file] Found {session_file_docs} verified documents. Sample IDs: {sample_ids}"
                    )

            except Exception as filter_err:
                logger.error(
                    f"[index_file] Error during document verification filtering: {filter_err}"
                )
                # Fallback to iterating if filter fails (less efficient)
                all_docs = (
                    self.document_store.filter_documents()
                )  # Get all and filter manually
                session_file_docs = 0
                for doc in all_docs:
                    if doc.meta.get("session_id") == session_id and str(
                        doc.meta.get("file_path")
                    ) == str(file_path):
                        session_file_docs += 1
                logger.info(
                    f"[index_file] Found {session_file_docs} documents via manual iteration during verification."
                )

            if (
                session_file_docs == 0 and documents_added > 0
            ):  # If we added docs but can't find them by filter
                logger.error(
                    f"[index_file] Verification failed: Added {documents_added} docs but found 0 for {file_path} in session {session_id}. Possible filter issue or write failure."
                )
                # Consider logging details of the documents that *were* added if possible
                return False
            elif session_file_docs < documents_added:
                logger.warning(
                    f"[index_file] Verification warning: Added {documents_added} docs but found only {session_file_docs} for {file_path} in session {session_id}."
                )
                # Decide if this is acceptable or should be a failure

            logger.info(
                f"[index_file] Successfully indexed file: {file_path} for session: {session_id} with {session_file_docs} documents verified in store."
            )
            return True

        except Exception as e:
            logger.error(f"[index_file] Error indexing file {file_path}: {str(e)}")
            logger.exception(
                "[index_file] Full traceback for index_file error:"
            )  # Ensure full traceback is logged
            return False

    def remove_file(self, file_path: str, session_id: str = "default") -> bool:
        """
        Remove an indexed file from the document store.

        Args:
            file_path: Path to the file to remove
            session_id: Session ID from which to remove the file

        Returns:
            bool: True if removal was successful, False otherwise
        """
        try:
            logger.info(f"Removing file: {file_path} from session: {session_id}")

            # Check if file was indexed in this session
            if (
                session_id in self.session_files
                and file_path in self.session_files[session_id]
            ):
                # Get all documents from this file and session
                all_docs = self.document_store.filter_documents()

                # Find documents that match this file path and session ID
                docs_to_remove = []
                file_name = os.path.basename(file_path)

                for doc in all_docs:
                    if (
                        doc.meta.get("file_path") == file_path
                        or doc.meta.get("file_name") == file_name
                    ) and doc.meta.get("session_id") == session_id:
                        docs_to_remove.append(doc.id)

                if docs_to_remove:
                    logger.info(f"Found {len(docs_to_remove)} documents to remove")
                    # Delete the documents
                    self.document_store.delete_documents(docs_to_remove)

                    # Remove from session tracking
                    self.session_files[session_id].remove(file_path)

                    logger.info(
                        f"Successfully removed file: {file_path} from session: {session_id}"
                    )
                    return True
                else:
                    logger.warning(
                        f"No documents found for file: {file_path} in session: {session_id}"
                    )
            else:
                logger.warning(f"File {file_path} not found in session: {session_id}")

            return False

        except Exception as e:
            logger.error(f"Error removing file {file_path}: {str(e)}")
            logger.exception("Full traceback:")
            return False

    def clear_session(self, session_id: str = "default") -> bool:
        """
        Clear all documents associated with a specific session.

        Args:
            session_id: Session ID to clear

        Returns:
            bool: True if clearing was successful, False otherwise
        """
        try:
            if session_id in self.session_files:
                logger.info(f"Clearing session: {session_id}")

                # Get all documents from this session
                all_docs = self.document_store.filter_documents()
                docs_to_remove = []

                for doc in all_docs:
                    if doc.meta.get("session_id") == session_id:
                        docs_to_remove.append(doc.id)

                if docs_to_remove:
                    logger.info(
                        f"Found {len(docs_to_remove)} documents to remove from session"
                    )
                    # Delete the documents
                    self.document_store.delete_documents(docs_to_remove)

                # Clear session tracking
                self.session_files[session_id] = set()

                logger.info(f"Successfully cleared session: {session_id}")
                return True
            else:
                logger.warning(f"Session {session_id} not found or already empty")
                return False

        except Exception as e:
            logger.error(f"Error clearing session {session_id}: {str(e)}")
            logger.exception("Full traceback:")
            return False

    def retrieve_context(
        self, query: str, session_id: str = "default"
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The user's query
            session_id: Session ID to retrieve documents from

        Returns:
            List[Document]: List of relevant documents
        """
        try:
            # Check if there are any documents in the store
            all_docs = self.document_store.filter_documents()

            # Filter for documents from this session
            session_docs = [
                doc for doc in all_docs if doc.meta.get("session_id") == session_id
            ]

            doc_count = len(session_docs)
            logger.info(
                f"Retrieving context for query: '{query}' in session {session_id}. Document count: {doc_count}"
            )

            if doc_count == 0:
                logger.warning(
                    f"No documents found for session {session_id}. Using model knowledge only."
                )
                return []

            # Special handling for SCBA-related queries to improve retrieval
            is_scba_query = False
            if any(
                term in query.lower()
                for term in [
                    "scba",
                    "breathing apparatus",
                    "doffing",
                    "donning",
                    "self-contained",
                ]
            ):
                is_scba_query = True
                logger.info(
                    "Detected SCBA-related query, applying specialized retrieval strategy"
                )
                # For SCBA queries, we'll combine results from both embedding and keyword-based search
                # with specialized keywords to boost recall

                # Add additional search terms for better matches
                expanded_query = query
                if "doffing" in query.lower() and "scba" in query.lower():
                    expanded_query = (
                        f"{query} remove taking off self-contained breathing apparatus"
                    )
                elif "donning" in query.lower() and "scba" in query.lower():
                    expanded_query = (
                        f"{query} put on wearing self-contained breathing apparatus"
                    )
                elif "scba" in query.lower():
                    expanded_query = f"{query} self-contained breathing apparatus"

                logger.info(f"Expanded query: '{expanded_query}'")
            else:
                expanded_query = query

            # Use a simplified approach that works for all queries
            # Focus on semantic similarity without special cases
            try:
                logger.info("Performing semantic retrieval for query...")
                embedded_query = self.text_embedder.run(text=expanded_query)
                embedding_shape = len(embedded_query["embedding"])
                logger.info(f"Query embedded successfully. Shape: {embedding_shape}")

                # Get documents from embedding retriever
                embedding_result = self.embedding_retriever.run(
                    query_embedding=embedded_query["embedding"],
                    top_k=self.top_k * 2,
                )
                embedding_docs = embedding_result["documents"]
                logger.info(
                    f"Embedding retrieval returned {len(embedding_docs)} documents"
                )

                # Also get BM25 results for keyword matching
                logger.info("Performing BM25 retrieval for query...")
                bm25_result = self.bm25_retriever.run(
                    query=expanded_query, top_k=self.top_k * 2
                )
                bm25_docs = bm25_result["documents"]
                logger.info(f"BM25 retriever returned {len(bm25_docs)} documents")

                # If it's an SCBA query, try additional keyword search with specific terms
                if is_scba_query:
                    logger.info("Performing additional SCBA-specific keyword search...")
                    scba_terms = [
                        "SCBA",
                        "self-contained breathing apparatus",
                        "doffing",
                        "donning",
                        "air pack",
                        "respirator",
                        "breathing protection",
                    ]
                    additional_docs = []

                    for term in scba_terms:
                        term_result = self.bm25_retriever.run(query=term, top_k=5)
                        additional_docs.extend(term_result["documents"])

                    logger.info(
                        f"SCBA-specific search returned {len(additional_docs)} additional documents"
                    )
                    # Combine with other results
                    bm25_docs.extend(additional_docs)

                # Combine results (deduplicating by document ID)
                seen_ids = set()
                all_docs = []

                # Add all documents to results, avoiding duplicates
                for doc in embedding_docs + bm25_docs:
                    if doc.id not in seen_ids:
                        all_docs.append(doc)
                        seen_ids.add(doc.id)

                logger.info(f"Combined unique documents: {len(all_docs)}")

                if not all_docs:
                    logger.warning("No documents retrieved from any method")
                    return []

                # Rerank all retrieved documents to get most relevant ones
                try:
                    reranker_result = self.reranker.run(documents=all_docs, query=query)
                    reranked_docs = reranker_result["documents"]
                    logger.info(f"Reranked to top {len(reranked_docs)} documents")

                    # Log reranked results
                    for i, doc in enumerate(reranked_docs[:3]):
                        logger.info(
                            f"Reranked Doc {i+1} score: {doc.score if hasattr(doc, 'score') else 'No score'}"
                        )
                        logger.info(
                            f"Reranked Doc {i+1} preview: {doc.content[:200]}..."
                        )

                    # For SCBA queries, ensure we have a good mix of documents
                    if is_scba_query and len(reranked_docs) > self.rerank_top_k:
                        # Keep all docs but take more than the usual limit
                        return reranked_docs[
                            : min(self.rerank_top_k + 5, len(reranked_docs))
                        ]

                    return reranked_docs[: self.rerank_top_k]
                except Exception as e:
                    logger.error(f"Reranking failed: {str(e)}")
                    logger.info("Using top documents without reranking")
                    return all_docs[: self.rerank_top_k]

            except Exception as e:
                logger.error(f"Retrieval error: {str(e)}")
                logger.exception("Retrieval error traceback:")
                return []

        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            logger.exception("Full traceback:")
            return []

    def _clean_model_output(self, answer: str) -> str:
        """
        Clean and format model output using Markdown principles.

        Args:
            answer: Raw model output (assumed to be mostly extracted answer)

        Returns:
            Cleaned and Markdown-formatted answer
        """
        if not answer or len(answer.strip()) < 10:
            logger.warning("Answer too short or empty, returning generic response")
            return "I apologize, but I'm having trouble generating a detailed response. Please try asking your question differently."

        # 1. Remove potential leading/trailing prompt remnants
        potential_prefixes = [
            "<|assistant|>",
            "assistant:",
            "response:",
            "answer:",
            "system:",
        ]
        for prefix in potential_prefixes:
            if answer.lower().startswith(prefix):
                answer = answer[len(prefix) :].lstrip(": \n")
                logger.info(f"Removed likely prompt prefix: '{prefix}'")
                break  # Stop after first match

        potential_suffixes = [
            "<|end|>",
            "<|endoftext|>",
            "<|EOS|>",
            "<|user|>",
        ]
        for suffix in potential_suffixes:
            if answer.lower().endswith(suffix):
                answer = answer[: -len(suffix)].rstrip(": \n")
                logger.info(f"Removed likely prompt suffix: '{suffix}'")
                break  # Stop after first match

        # 2. Basic Whitespace Normalization
        answer = re.sub(r"[ \t]+", " ", answer)  # Normalize horizontal whitespace
        answer = answer.strip()  # Remove leading/trailing whitespace
        answer = re.sub(
            r"\n{3,}", "\n\n", answer
        )  # Replace 3+ newlines with exactly two

        # 3. Markdown Formatting Logic
        lines = answer.split("\n")
        formatted_lines = []
        in_list = False
        current_list_type = None  # 'numbered' or 'bullet'
        expected_number = 1

        # Process each line for Markdown formatting
        for line in lines:
            line = line.strip()
            if not line:
                # Empty line - preserve for paragraph breaks
                formatted_lines.append("")
                # Empty line typically ends a list
                if in_list:
                    in_list = False
                    current_list_type = None
                    expected_number = 1
                continue

            # Check for numbered list items
            numbered_match = re.match(r"^(\d+)[\.\)\s\-]+\s*(.+)$", line)
            if numbered_match:
                number = int(numbered_match.group(1))
                content = numbered_match.group(2).strip()

                # Handle numbered list formatting
                if (
                    not in_list
                    or current_list_type != "numbered"
                    or number == expected_number
                ):
                    # Normal case - expected sequence or new list
                    in_list = True
                    current_list_type = "numbered"
                    expected_number = number + 1
                    formatted_lines.append(f"{number}. {content}")
                else:
                    # Handle incorrect numbering
                    if number == 1:
                        # Start of new list
                        in_list = True
                        current_list_type = "numbered"
                        expected_number = 2
                        formatted_lines.append(f"1. {content}")
                    else:
                        # Continue with corrected numbering
                        formatted_lines.append(f"{expected_number}. {content}")
                        expected_number += 1
                continue

            # Check for bullet list items
            bullet_match = re.match(r"^[•*\-+]+\s*(.+)$", line)
            if bullet_match:
                content = bullet_match.group(1).strip()

                # Format as proper Markdown bullet list
                in_list = True
                current_list_type = "bullet"
                formatted_lines.append(f"* {content}")
                continue

            # Handle regular text/paragraphs
            if in_list:
                # Check if this might be a list item continuation
                if formatted_lines and (line[0].islower() or line[0] in "([,;:'"):
                    # Append to previous line as continuation
                    formatted_lines[-1] += f" {line}"
                else:
                    # New paragraph ends the list
                    in_list = False
                    current_list_type = None
                    expected_number = 1
                    formatted_lines.append(line)
            else:
                # Regular paragraph text
                formatted_lines.append(line)

        # 4. Final Formatting and Cleanup
        formatted_answer = "\n\n".join(
            [l.strip() for l in " ".join(formatted_lines).split("\n\n")]
        )

        # Ensure proper list formatting
        # Single newline between list items of the same type
        formatted_answer = re.sub(
            r"(\d+\.\s[^\n]+)\n\n(\d+\.)", r"\1\n\2", formatted_answer
        )
        formatted_answer = re.sub(
            r"(\*\s[^\n]+)\n\n(\*\s)", r"\1\n\2", formatted_answer
        )

        # Ensure proper paragraph and list separation
        # Force new line for list items that appear mid-paragraph
        formatted_answer = re.sub(
            r"([^\n])([\d]+\.|\*\s)", r"\1\n\n\2", formatted_answer
        )
        # Make sure ordered lists have proper formatting (number, period, space)
        formatted_answer = re.sub(r"(\d+)[^\.\s](\s)", r"\1.\2", formatted_answer)
        # Ensure paragraphs have double line breaks
        formatted_answer = re.sub(r"([^\n])\n([^\n])", r"\1\n\n\2", formatted_answer)

        # Normalize line endings
        formatted_answer = formatted_answer.replace("\r\n", "\n").replace("\r", "\n")

        # Fix for list output appearing as one blob - more explicit processing
        # First, identify the numbered lists in the text
        list_pattern = r"(\d+\.\s.+?)(?=\n\d+\.|$)"
        matches = re.findall(list_pattern, formatted_answer, re.DOTALL)

        # If we found numbered lists
        if matches:
            # Split the formatted answer by the lists
            segments = re.split(list_pattern, formatted_answer, flags=re.DOTALL)

            # Rebuild with proper line breaks
            new_answer = ""
            for i in range(len(segments)):
                if i % 2 == 0:  # Regular text segments
                    new_answer += segments[i]
                else:  # List item segments
                    # Add each list item with proper line break
                    item_text = segments[i].strip()
                    # Ensure item starts with a number followed by period and space
                    if not re.match(r"^\d+\.\s", item_text):
                        item_number = re.match(r"^\d+", item_text)
                        if item_number:
                            item_text = f"{item_number.group(0)}. {item_text[len(item_number.group(0)):].lstrip('.) ')}"

                    # Add the list item with a guaranteed newline before it
                    if not new_answer.endswith("\n\n"):
                        new_answer += "\n\n"
                    new_answer += item_text + "\n"

            formatted_answer = new_answer.strip()

            # Now make sure multi-line lists have proper indentation and spacing
            formatted_answer = re.sub(
                r"(\d+\.\s.+?)(\n)(?!\d+\.|\s*$|\n)", r"\1\2    ", formatted_answer
            )

        # Process bullet lists with asterisks
        bullet_pattern = r"(\*\s.+?)(?=\n\*\s|$)"
        bullet_matches = re.findall(bullet_pattern, formatted_answer, re.DOTALL)

        # If we found bullet lists
        if bullet_matches:
            # Split the formatted answer by the bullet points
            segments = re.split(bullet_pattern, formatted_answer, flags=re.DOTALL)

            # Rebuild with proper line breaks
            new_answer = ""
            for i in range(len(segments)):
                if i % 2 == 0:  # Regular text segments
                    new_answer += segments[i]
                else:  # Bullet item segments
                    # Add each bullet item with proper line break
                    item_text = segments[i].strip()

                    # Ensure item starts with an asterisk and space
                    if not re.match(r"^\*\s", item_text):
                        item_text = f"* {item_text.lstrip('*').lstrip()}"

                    # Add the bullet item with a guaranteed newline before it
                    if not new_answer.endswith("\n\n"):
                        new_answer += "\n\n"
                    new_answer += item_text + "\n"

            formatted_answer = new_answer.strip()

            # Now make sure multi-line bullet lists have proper indentation and spacing
            formatted_answer = re.sub(
                r"(\*\s.+?)(\n)(?!\*\s|\d+\.|\s*$|\n)", r"\1\2    ", formatted_answer
            )

        return formatted_answer

    def generate_response(
        self,
        query: str,
        context_docs: Optional[List[Document]] = None,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Generate a response for the given query using the provided context.

        Args:
            query: The user's query
            context_docs: Optional list of context documents
            session_id: Session ID for context retrieval

        Returns:
            Dict containing the response and context information
        """
        try:
            logger.info(
                f"Generating response for query: '{query}' in session {session_id}"
            )

            # Check if query is asking for specific data that needs documents
            specific_data_patterns = [
                # General statistics and data patterns
                r"percentage",
                r"percent",
                r"how many",
                r"what percent",
                r"stats",
                r"statistics",
                r"survey",
                r"report",
                r"data",
                r"what number",
                r"how much",
                r"\d{4} survey",
                r"spring \d{4}",
                r"annual report",
                r"\d{4}-\d{4}",
                r"\d{4} report",
                # Patterns for specific numerical queries
                r"how many people",
                r"how many participants",
                r"how many respondents",
                r"what percentage of",
                r"what fraction of",
                r"how much funding",
                # Budget and financial patterns
                r"budget",
                r"funding",
                r"financial",
                r"cost",
                r"expense",
                # Time-specific patterns
                r"in \d{4}",
                r"during \d{4}",
                r"last year",
                r"this year",
                r"next year",
                # Comparison patterns
                r"compared to",
                r"versus",
                r"vs",
                r"difference between",
                r"changes? from",
                # Opinion/survey patterns
                r"opinion",
                r"feedback",
                r"responded",
                r"respondents",
                r"surveyed",
                r"participants said",
                r"residents indicated",
            ]

            is_specific_data_query = any(
                re.search(pattern, query.lower()) for pattern in specific_data_patterns
            )

            # Check if there are any documents in the session
            session_has_docs = False
            for doc in self.document_store.filter_documents():
                if doc.meta.get("session_id") == session_id:
                    session_has_docs = True
                    break

            # Retrieve context if not provided
            if context_docs is None:
                context_docs = self.retrieve_context(query, session_id=session_id)
                logger.info(
                    f"Retrieved {len(context_docs)} context documents for session {session_id}"
                )

            # DEBUG: Log detailed info about retrieved context
            if context_docs:
                logger.info("================ DETAILED CONTEXT DEBUG ================")
                logger.info(f"Total context documents retrieved: {len(context_docs)}")
                for i, doc in enumerate(context_docs):
                    logger.info(f"Document {i+1}:")
                    logger.info(f"  Content length: {len(doc.content)}")
                    logger.info(f"  Content preview: {doc.content[:200]}...")
                    if hasattr(doc, "score"):
                        logger.info(f"  Relevance score: {doc.score}")
                    logger.info(f"  Metadata: {doc.meta}")
                    if self._detect_garbled_text(doc.content):
                        logger.info(f"  GARBLED TEXT DETECTED in document {i+1}")
                logger.info("=====================================================")

            # Check if we have valid context documents
            valid_context = False
            if context_docs:
                # Verify content is not garbled
                garbled_count = 0
                for doc in context_docs:
                    if self._detect_garbled_text(doc.content):
                        garbled_count += 1

                # If too many documents are garbled, consider context invalid
                if garbled_count / len(context_docs) > 0.7:  # More than 70% garbled
                    logger.warning(
                        f"{garbled_count} out of {len(context_docs)} documents contain garbled text"
                    )
                    valid_context = False
                else:
                    valid_context = True

            # Special case: specific data query with files but no relevant context
            if (
                is_specific_data_query
                and session_has_docs
                and (not context_docs or not valid_context)
            ):
                logger.warning(
                    "Specific data query with documents available but no relevant context found"
                )
                answer = "I couldn't find the specific information you're asking about in the uploaded documents. The data about percentages, survey results, or statistics you're looking for might not be present in these documents, or may be in a different format. You may want to try a different query or upload additional relevant documents."

                return {
                    "query": query,
                    "answer": answer,
                    "context": [
                        {
                            "source": "No relevant data found",
                            "content": "The specific data requested was not found in the uploaded documents.",
                            "file_name": "No matching files",
                        }
                    ],
                }

            if not context_docs or not valid_context:
                logger.warning("No valid context documents retrieved")
                # Generate a response with special notice about document issues
                if context_docs and not valid_context:
                    return self.generate_from_knowledge(
                        query, document_issue_notice=True
                    )
                else:
                    return self.generate_from_knowledge(query)

            logger.info(f"Number of context documents: {len(context_docs)}")

            if context_docs:
                logger.info("Context document details:")
                for i, doc in enumerate(context_docs[:3]):  # Log first 3 docs
                    logger.info(f"Doc {i+1}:")
                    logger.info(f"  Content length: {len(doc.content)}")
                    logger.info(f"  Content preview: {doc.content[:200]}...")
                    logger.info(f"  Metadata: {doc.meta}")

            # Create context string from documents
            context_str = self._format_context_for_prompt(context_docs)

            # If the context string indicates no readable content, fall back to model knowledge
            if context_str.startswith("No readable content could be extracted"):
                logger.warning(
                    "Context formatting indicates no usable content, falling back to model knowledge"
                )
                return self.generate_from_knowledge(query, document_issue_notice=True)

            # Log the context string length
            logger.info(f"Context string length: {len(context_str)} chars")

            # Generate prompt based on model type
            prompt = self._create_prompt_with_context(query, context_str)

            # Truncate prompt if necessary
            if len(prompt) > 8000:  # Conservative limit for most models
                logger.warning(f"Prompt too long ({len(prompt)} chars), truncating")
                prompt = self._truncate_prompt(prompt, 8000)

            # DEBUG: Log the full prompt
            logger.info("================ FULL PROMPT DEBUG ================")
            logger.info(f"Final prompt length: {len(prompt)} chars")
            logger.info(f"Query length: {len(query)} chars")
            logger.info(f"Context string length: {len(context_str)} chars")
            logger.info(f"MODEL PROMPT:\n{prompt}")
            logger.info("==================================================")

            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            token_count = len(inputs["input_ids"][0])
            logger.info(f"Tokenized input length: {token_count} tokens")

            # Generate the answer with improved parameters for better quality
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,  # Increased for Llama 2
                    min_new_tokens=100,
                    temperature=0.55,  # Slightly reduced for more focused outputs
                    top_p=0.95,  # Increased for Llama 2
                    top_k=60,  # Increased for Llama 2
                    repetition_penalty=1.15,  # Reduced for Llama 2 (less needed)
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            # >>> ADDED DEBUG LOG: Raw output tensor
            logger.info(f"DEBUG_GARBLE: Raw output tensor shape: {outputs[0].shape}")
            logger.info(
                f"DEBUG_GARBLE: Raw output tensor (first 50 tokens): {outputs[0][:50]}"
            )
            logger.info(
                f"DEBUG_GARBLE: Raw output tensor (last 50 tokens): {outputs[0][-50:]}"
            )

            # Try both decoding approaches: with and without special tokens
            full_output_with_special = self.tokenizer.decode(
                outputs[0], skip_special_tokens=False
            )
            # >>> ADDED DEBUG LOG: After decode (with special)
            logger.info(
                f"DEBUG_GARBLE: Decoded (with special) - First 100 chars: {full_output_with_special[:100]}"
            )
            logger.info(
                f"DEBUG_GARBLE: Decoded (with special) - Last 100 chars: {full_output_with_special[-100:]}"
            )

            full_output_no_special = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            # >>> ADDED DEBUG LOG: After decode (no special)
            logger.info(
                f"DEBUG_GARBLE: Decoded (no special) - First 100 chars: {full_output_no_special[:100]}"
            )
            logger.info(
                f"DEBUG_GARBLE: Decoded (no special) - Last 100 chars: {full_output_no_special[-100:]}"
            )

            # Debug log the raw outputs with special tokens visible
            logger.info("================ RAW MODEL OUTPUT ================")
            logger.info(
                f"Output length (with special tokens): {len(full_output_with_special)} chars"
            )
            logger.info(
                f"Output length (no special tokens): {len(full_output_no_special)} chars"
            )

            # Log if common markers are found
            markers = ["<|assistant|>", "<|end|>", "<|endoftext|>", " { "]
            for marker in markers:
                if marker in full_output_with_special:
                    logger.info(
                        f"Found marker '{marker}' at position {full_output_with_special.find(marker)}"
                    )

            # Log a truncated version of the outputs if they're very long
            if len(full_output_with_special) > 1000:
                logger.info(
                    f"Raw output with special tokens (first 500 chars): {full_output_with_special[:500]}"
                )
                logger.info(
                    f"Raw output with special tokens (last 500 chars): {full_output_with_special[-500:]}"
                )
            else:
                logger.info(
                    f"Raw output with special tokens: {full_output_with_special}"
                )

            logger.info(
                f"Raw output without special tokens: {full_output_no_special[:500]}"
            )
            logger.info("=================================================")

            # First try to extract from the output with special tokens
            answer = self._extract_answer_from_output(full_output_with_special, prompt)

            # If the answer seems garbled or empty, try the version without special tokens
            if not answer or self._detect_garbled_text(answer, is_model_output=True):
                logger.info(
                    "Extraction from output with special tokens failed, trying without special tokens"
                )
                # For the no-special-tokens version, we'll need to manually extract after the prompt
                # Since we don't have markers to help us
                if prompt in full_output_no_special:
                    alt_answer = full_output_no_special.split(prompt, 1)[1].strip()
                else:
                    # If prompt not found, just use the whole output
                    alt_answer = full_output_no_special

                # If the alternative answer is substantially better, use it
                if alt_answer and len(alt_answer) > len(answer) * 2:
                    logger.info("Using output without special tokens instead")
                    answer = alt_answer

            # Clean and post-process the answer
            answer = self._clean_model_output(answer)
            # >>> ADDED DEBUG LOG: Before final cleaning
            logger.info(
                f"DEBUG_GARBLE: Text before _clean_model_output - First 100 chars: {answer[:100]}"
            )
            logger.info(
                f"DEBUG_GARBLE: Text before _clean_model_output - Last 100 chars: {answer[-100:]}"
            )

            # Special check for knowledge-only responses with line breaking issues
            if (
                answer.count("\n") > answer.count(".") * 0.7
            ):  # Many newlines relative to sentences
                logger.info(
                    "Detected excessive line breaks in knowledge-only response, applying extra formatting"
                )

                # Don't reformat if the response contains Markdown list markers
                if re.search(
                    r"(\n\s*\d+\.\s|\n\s*\*\s|\n\s*-\s|\d+\.\s|\*\s|\n\d+\.|\n\s*\d+\))",
                    answer,
                ):
                    logger.info("Detected Markdown lists, preserving formatting")
                else:
                    # More aggressive newline removal for knowledge-only mode
                    answer = re.sub(
                        r"\n+", " ", answer
                    )  # Replace all newlines with spaces
                    answer = re.sub(r"\s{2,}", " ", answer)  # Normalize whitespace

                    # Re-introduce paragraph breaks at proper sentence boundaries
                    sentences = re.split(r"(?<=[.!?])\s+", answer)
                    paragraphs = []
                    current_para = []

                    for sentence in sentences:
                        if not sentence.strip():
                            continue

                        current_para.append(sentence)
                        # Start new paragraph every 2-3 sentences
                        if len(current_para) >= 2 and random.random() < 0.4:
                            paragraphs.append(" ".join(current_para))
                            current_para = []

                    # Add any remaining sentences
                    if current_para:
                        paragraphs.append(" ".join(current_para))

                    # Join paragraphs with newlines
                    answer = "\n\n".join(paragraphs)
                    logger.info(
                        f"Reformatted response into {len(paragraphs)} paragraphs"
                    )

            # Final check for list items that should be properly formatted
            # Check if there are numbered list items without proper formatting
            list_check_pattern = r"(?:^|\n)[ \t]*(\d+)[ \t]*[\)\.\-][ \t]*(.+?)(?:$|\n)"
            if re.search(list_check_pattern, answer, re.MULTILINE):
                logger.info(
                    "Found potentially unformatted numbered list items, applying Markdown formatting"
                )
                # Reformat numbered lists with proper Markdown
                answer = re.sub(
                    list_check_pattern, r"\n\1. \2\n", answer, flags=re.MULTILINE
                )
                # Normalize whitespace and newlines
                answer = re.sub(r"[ \t]+", " ", answer)
                answer = re.sub(r"\n{3,}", "\n\n", answer)

            # Check for bullet points
            bullet_check_pattern = r"(?:^|\n)[ \t]*[\*\-\•\+][ \t]*(.+?)(?:$|\n)"
            if re.search(bullet_check_pattern, answer, re.MULTILINE):
                logger.info(
                    "Found potentially unformatted bullet list items, applying Markdown formatting"
                )
                # Reformat bullet lists with proper Markdown
                answer = re.sub(
                    bullet_check_pattern, r"\n* \1\n", answer, flags=re.MULTILINE
                )
                # Normalize whitespace and newlines
                answer = re.sub(r"[ \t]+", " ", answer)
                answer = re.sub(r"\n{3,}", "\n\n", answer)

            return {
                "query": query,
                "answer": answer,
                "context": self._prepare_context_metadata(context_docs),
            }

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            logger.exception("Full traceback:")
            return {
                "query": query,
                "answer": "I apologize, but an internal error occurred while processing your request. Please try again or rephrase your question.",
                "context": [
                    {
                        "source": "Internal Error",
                        "content": "An error occurred while processing your request.",
                        "file_name": "Error",
                    }
                ],
            }

    def generate_from_knowledge(
        self, query: str, document_issue_notice: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a response using only the model's internal knowledge.
        Used when no relevant context is found or documents have issues.

        Args:
            query: The user's query
            document_issue_notice: If True, add a notice about document issues.

        Returns:
            Dict containing the response and context information (indicating no context used).
        """
        # >>> ADDED DEBUG LOG: Start of generate_from_knowledge
        logger.info(
            f"DEBUG_GARBLE: Entering generate_from_knowledge for query: {query}"
        )
        try:
            logger.info(
                f"Generating response for query '{query}' using model knowledge only."
            )

            # Manually create prompt without context
            system_message = "You are a knowledgeable first responder assistant. Answer the question based on your general knowledge."
            if document_issue_notice:
                system_message += "\nNote: There were issues reading provided documents, so this answer is based on general knowledge."

            # Add specific instruction about formatting to prevent fragmented responses
            system_message += "\nPlease provide a clear, complete answer in full sentences and well-structured paragraphs. Avoid excessive line breaks or fragments."

            # Add instruction to format lists as Markdown
            system_message += "\nFor any lists in your response, use proper Markdown formatting: numbered lists with '1. ', '2. ', etc., and bullet lists with '* ' at the start of each item."

            prompt = f"<|system|>\n{system_message}\n<|user|>\n{query}\n<|assistant|>"

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            token_count = len(inputs["input_ids"][0])
            logger.info(f"Knowledge-only prompt token count: {token_count}")

            # Generate with adjusted parameters to reduce text fragmentation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=400,
                    min_new_tokens=50,
                    temperature=0.6,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            # >>> ADDED DEBUG LOG: Raw output tensor (knowledge mode)
            logger.info(
                f"DEBUG_GARBLE: [Knowledge] Raw output tensor shape: {outputs[0].shape}"
            )
            logger.info(
                f"DEBUG_GARBLE: [Knowledge] Raw output tensor (first 50 tokens): {outputs[0][:50]}"
            )
            logger.info(
                f"DEBUG_GARBLE: [Knowledge] Raw output tensor (last 50 tokens): {outputs[0][-50:]}"
            )

            # Decode with both approaches
            full_output_with_special = self.tokenizer.decode(
                outputs[0], skip_special_tokens=False
            )
            # >>> ADDED DEBUG LOG: After decode (with special) (knowledge mode)
            logger.info(
                f"DEBUG_GARBLE: [Knowledge] Decoded (with special) - First 100 chars: {full_output_with_special[:100]}"
            )
            logger.info(
                f"DEBUG_GARBLE: [Knowledge] Decoded (with special) - Last 100 chars: {full_output_with_special[-100:]}"
            )

            full_output_no_special = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            # >>> ADDED DEBUG LOG: After decode (no special) (knowledge mode)
            logger.info(
                f"DEBUG_GARBLE: [Knowledge] Decoded (no special) - First 100 chars: {full_output_no_special[:100]}"
            )
            logger.info(
                f"DEBUG_GARBLE: [Knowledge] Decoded (no special) - Last 100 chars: {full_output_no_special[-100:]}"
            )

            # Debug log the raw output with special tokens visible
            logger.info(
                "================ RAW MODEL OUTPUT (KNOWLEDGE MODE) ================"
            )
            logger.info(f"Output length: {len(full_output_with_special)} chars")
            # Log if common markers are found
            markers = ["<|assistant|>", "<|end|>", "<|endoftext|>", " { "]
            for marker in markers:
                if marker in full_output_with_special:
                    logger.info(
                        f"Found marker '{marker}' at position {full_output_with_special.find(marker)}"
                    )
            # Log a truncated version of the output if it's very long
            if len(full_output_with_special) > 1000:
                logger.info(
                    f"Raw output (first 500 chars): {full_output_with_special[:500]}"
                )
                logger.info(
                    f"Raw output (last 500 chars): {full_output_with_special[-500:]}"
                )
            else:
                logger.info(f"Raw output: {full_output_with_special}")
            logger.info(
                "================================================================="
            )

            # First try to extract from output with special tokens
            answer = self._extract_answer_from_output(full_output_with_special, prompt)

            # If the extracted answer seems problematic, try the version without special tokens
            # >>> MODIFYING CONDITION: Removed newline check to prevent incorrect fallback <<<<
            if (
                not answer
                or len(answer.strip()) < 20
                # or answer.count("\n") > answer.count(".") # Removed this condition
            ):
                logger.info(
                    "Initial extraction problematic (empty/short), trying alternative extraction without special tokens"
                )
                # Get everything after the prompt
                if prompt in full_output_no_special:
                    alt_answer = full_output_no_special.split(prompt, 1)[1].strip()
                else:
                    alt_answer = full_output_no_special

                # If alternative answer seems better, use it
                if len(alt_answer) > len(answer):
                    logger.info(
                        "Using no-special-tokens extraction as it produced more content"
                    )
                    answer = alt_answer

            # Clean and post-process the answer
            answer = self._clean_model_output(answer)
            # >>> ADDED DEBUG LOG: Before final cleaning (knowledge mode)
            logger.info(
                f"DEBUG_GARBLE: [Knowledge] Text before _clean_model_output - First 100 chars: {answer[:100]}"
            )
            logger.info(
                f"DEBUG_GARBLE: [Knowledge] Text before _clean_model_output - Last 100 chars: {answer[-100:]}"
            )

            # Special check for knowledge-only responses with line breaking issues
            if (
                answer.count("\n") > answer.count(".") * 0.7
            ):  # Many newlines relative to sentences
                logger.info(
                    "Detected excessive line breaks in knowledge-only response, applying extra formatting"
                )

                # Don't reformat if the response contains Markdown list markers
                if re.search(
                    r"(\n\s*\d+\.\s|\n\s*\*\s|\n\s*-\s|\d+\.\s|\*\s|\n\d+\.|\n\s*\d+\))",
                    answer,
                ):
                    logger.info("Detected Markdown lists, preserving formatting")
                else:
                    # More aggressive newline removal for knowledge-only mode
                    answer = re.sub(
                        r"\n+", " ", answer
                    )  # Replace all newlines with spaces
                    answer = re.sub(r"\s{2,}", " ", answer)  # Normalize whitespace

                    # Re-introduce paragraph breaks at proper sentence boundaries
                    sentences = re.split(r"(?<=[.!?])\s+", answer)
                    paragraphs = []
                    current_para = []

                    for sentence in sentences:
                        if not sentence.strip():
                            continue

                        current_para.append(sentence)
                        # Start new paragraph every 2-3 sentences
                        if len(current_para) >= 2 and random.random() < 0.4:
                            paragraphs.append(" ".join(current_para))
                            current_para = []

                    # Add any remaining sentences
                    if current_para:
                        paragraphs.append(" ".join(current_para))

                    # Join paragraphs with newlines
                    answer = "\n\n".join(paragraphs)
                    logger.info(
                        f"Reformatted response into {len(paragraphs)} paragraphs"
                    )

            # Final check for list items that should be properly formatted
            # Check if there are numbered list items without proper formatting
            list_check_pattern = r"(?:^|\n)[ \t]*(\d+)[ \t]*[\)\.\-][ \t]*(.+?)(?:$|\n)"
            if re.search(list_check_pattern, answer, re.MULTILINE):
                logger.info(
                    "Found potentially unformatted numbered list items, applying Markdown formatting"
                )
                # Reformat numbered lists with proper Markdown
                answer = re.sub(
                    list_check_pattern, r"\n\1. \2\n", answer, flags=re.MULTILINE
                )
                # Normalize whitespace and newlines
                answer = re.sub(r"[ \t]+", " ", answer)
                answer = re.sub(r"\n{3,}", "\n\n", answer)

            # Check for bullet points
            bullet_check_pattern = r"(?:^|\n)[ \t]*[\*\-\•\+][ \t]*(.+?)(?:$|\n)"
            if re.search(bullet_check_pattern, answer, re.MULTILINE):
                logger.info(
                    "Found potentially unformatted bullet list items, applying Markdown formatting"
                )
                # Reformat bullet lists with proper Markdown
                answer = re.sub(
                    bullet_check_pattern, r"\n* \1\n", answer, flags=re.MULTILINE
                )
                # Normalize whitespace and newlines
                answer = re.sub(r"[ \t]+", " ", answer)
                answer = re.sub(r"\n{3,}", "\n\n", answer)

            return {
                "query": query,
                "answer": answer,
                "context": [
                    {
                        "source": "Model Knowledge",
                        "content": "No relevant documents were found or used for this response.",
                        "file_name": "N/A",
                    }
                ],
            }
        except Exception as e:
            logger.error(f"Error generating response from knowledge: {str(e)}")
            logger.exception("Full traceback:")
            # Return the generic error response structure but indicate the source of error
            return {
                "query": query,
                "answer": "I apologize, but an internal error occurred while generating the response. Please try again.",
                "context": [
                    {
                        "source": "Internal Error",
                        "content": "An error occurred during response generation.",
                        "file_name": "Error",
                    }
                ],
            }
        # >>> ADDED DEBUG LOG: Exiting generate_from_knowledge

    def _format_context_for_prompt(self, context_docs: List[Document]) -> str:
        """Format the context documents into a string for the prompt."""
        context_str = ""
        garbled_docs_count = 0
        usable_docs_count = 0

        for i, doc in enumerate(context_docs):
            # Include document number and file source in the context
            file_name = doc.meta.get("file_name", "Unknown source")

            # Clean the content once more before including it
            content = doc.content

            # Check for garbled text using the helper method
            if self._detect_garbled_text(content):
                garbled_docs_count += 1
                logger.warning(
                    f"Document {i+1} appears to contain garbled text. Applying aggressive cleaning."
                )

                # Apply more aggressive cleaning for inclusion in context
                # Replace non-ASCII characters with spaces
                # Remove excessive punctuation that might be encoding artifacts
                content = re.sub(r"[^\w\s\.,;:!?()-]{2,}", " ", content)
                # Add spaces between lowercase and uppercase letters to fix concatenated words
                content = re.sub(r"([a-z])([A-Z])", r"\1 \2", content)
                # Normalize whitespace
                content = re.sub(r"\s+", " ", content).strip()

                # If content is too short after cleaning, skip this document
                if len(content.strip()) < 20:
                    logger.warning(
                        f"Document {i+1} was too short after cleaning and will be excluded"
                    )
                    continue

                # Check if content is still garbled after cleaning
                if self._detect_garbled_text(content):
                    logger.warning(
                        f"Document {i+1} is still garbled after cleaning, skipping"
                    )
                    continue

            # Add the document to the context string
            context_str += f"Document {i+1} (from {file_name}):\n{content}\n\n"
            usable_docs_count += 1

        # Log statistics about garbled documents
        if garbled_docs_count > 0:
            logger.warning(
                f"Found {garbled_docs_count} documents with potentially garbled text out of {len(context_docs)}"
            )

        # If all documents were garbled/skipped, provide a notice
        if not context_str.strip():
            return "No readable content could be extracted from the documents due to possible encoding or formatting issues."

        # If very few usable documents from a larger set, add a note
        if usable_docs_count < len(context_docs) * 0.3 and len(context_docs) > 3:
            context_str += "\nNote: Many documents were excluded due to formatting or encoding issues. Results may be limited.\n\n"

        # Trim if too long
        if len(context_str) > 6000:
            logger.warning(
                f"Context too long ({len(context_str)} chars), trimming to 6000 chars"
            )
            context_str = context_str[:6000] + "..."

        return context_str

    def _create_prompt_with_context(self, query: str, context_str: str) -> str:
        """Create a properly formatted prompt for Phi-4 with context using manual construction."""
        system_message = (
            "You are a knowledgeable first responder assistant. Use the provided context to answer the question accurately. "
            "If the context doesn't contain the answer, state that clearly.\n"
            "Guidelines:\n"
            "1. Base answers strictly on the provided context.\n"
            "2. Be concise and precise.\n"
            "3. If unsure or the answer isn't in the context, say so.\n"
            "4. Prioritize safety information."
        )
        # Ensure context string is correctly handled
        context_section = (
            f"<|context|>{context_str}<|endofcontext|>\n\n" if context_str else ""
        )

        # Manually construct the full prompt
        prompt = (
            f"<|system|>\n{system_message}\n{context_section}"
            f"<|user|>\n{query}\n<|assistant|>"
        )
        return prompt

    def _extract_answer_from_output(self, full_output: str, prompt: str) -> str:
        """Extract just the assistant's response from the full output, primarily looking for the marker."""
        # >>> ADDED DEBUG LOG: Entering _extract_answer_from_output
        logger.info(
            f"DEBUG_GARBLE: Entering _extract_answer_from_output - Input First 100: '{full_output[:100]}'"
        )
        logger.info(
            f"DEBUG_GARBLE: Entering _extract_answer_from_output - Input Last 100: '{full_output[-100:]}'"
        )

        assistant_marker = "<|assistant|>"
        # Add variations sometimes seen
        assistant_markers = [assistant_marker, "assistant:", "<|im_start|>assistant"]
        end_markers = [
            "<|end|>",
            "<|endoftext|>",
            "<|EOS|>",
            " { ",
            "<|user|>",
            "user:",
            "<|im_end|>",
        ]

        # Normalize output slightly for more reliable marker detection
        normalized_output = full_output.strip()

        # DEBUG: Log raw output characteristics
        logger.info(f"DEBUG: Raw output length: {len(normalized_output)} characters")
        logger.info(f"DEBUG: Raw output first 50 chars: '{normalized_output[:50]}'")
        logger.info(f"DEBUG: Raw output last 50 chars: '{normalized_output[-50:]}'")

        found_marker = None
        last_marker_pos = -1

        # Primary method: Find the *last* occurrence of any known assistant marker
        for marker in assistant_markers:
            pos = normalized_output.rfind(marker)
            if pos > last_marker_pos:
                last_marker_pos = pos
                found_marker = marker

        if last_marker_pos != -1 and found_marker:
            # Get everything after the found assistant marker
            response = normalized_output[last_marker_pos + len(found_marker) :].strip()
            logger.info(
                f"DEBUG: Found assistant marker '{found_marker}' at position {last_marker_pos}"
            )
            logger.info(
                f"DEBUG: Extracted after marker (first 50 chars): '{response[:50]}'...\"\nDEBUG: (last 50 chars): ...'{response[-50:]}'"
            )

            # Remove any end markers appearing *after* the start of the response
            earliest_end_marker_pos = len(response)
            found_end_marker = None
            for end_marker in end_markers:
                pos = response.find(end_marker)
                if pos != -1 and pos < earliest_end_marker_pos:
                    earliest_end_marker_pos = pos
                    found_end_marker = end_marker

            if found_end_marker:
                logger.info(
                    f"DEBUG: Found end marker '{found_end_marker}' at position {earliest_end_marker_pos} in extracted response"
                )
                response = response[:earliest_end_marker_pos].strip()
                logger.info(
                    f"DEBUG: Response after removing end marker (first 50 chars): '{response[:50]}'...\"\nDEBUG: (last 50 chars): ...'{response[-50:]}'"
                )

            # Check if the response is just EOS or empty after stripping markers
            if response and response.lower() != self.tokenizer.eos_token.lower():
                # Early cleaning of response to handle leading punctuation often left by markers
                response = response.lstrip(":\\n ")
                logger.info("Extracted response using assistant marker.")
                logger.info(
                    f"DEBUG: Final extracted response (first 50 chars): '{response[:50]}'...\"\nDEBUG: (last 50 chars): ...'{response[-50:]}'"
                )
                return response
            else:
                logger.warning(
                    "Found assistant marker but response was empty or EOS after stripping end markers."
                )
                logger.info(
                    f"DEBUG: Empty response after marker processing, response: '{response}'"
                )
        else:
            logger.warning(
                f"Could not find any known assistant markers: {assistant_markers}"
            )

        # Fallback 1: If marker method failed, try stripping the prompt (less reliable)
        # Make prompt stripping more robust by checking variations
        normalized_prompt = prompt.strip()
        if normalized_output.startswith(normalized_prompt):
            stripped_response = normalized_output[len(normalized_prompt) :].strip()
            logger.info(
                f"DEBUG: Fallback 1 - prompt-stripped response (first 50 chars): '{stripped_response[:50]}'...\"\nDEBUG: (last 50 chars): ...'{stripped_response[-50:]}'"
            )

            # Also check for end markers in the stripped response
            earliest_end_marker_pos = len(stripped_response)
            found_end_marker = None
            for end_marker in end_markers:
                pos = stripped_response.find(end_marker)
                if pos != -1 and pos < earliest_end_marker_pos:
                    earliest_end_marker_pos = pos
                    found_end_marker = end_marker

            if found_end_marker:
                logger.info(
                    f"DEBUG: Fallback 1 - Found end marker '{found_end_marker}' at position {earliest_end_marker_pos}"
                )
                stripped_response = stripped_response[:earliest_end_marker_pos].strip()

            if (
                stripped_response
                and stripped_response.lower() != self.tokenizer.eos_token.lower()
            ):
                stripped_response = stripped_response.lstrip(
                    ":\\n "
                )  # Clean potential leading chars
                logger.warning("Used prompt stripping as fallback for extraction.")
                logger.info(
                    f"DEBUG: Fallback 1 final response (first 50 chars): '{stripped_response[:50]}'...\"\nDEBUG: (last 50 chars): ...'{stripped_response[-50:]}'"
                )
                return stripped_response

        # Fallback 2: Check if the *entire* output seems to be the answer (no prompt structure)
        # This might happen if the model completely ignored the prompt format
        is_likely_answer_only = True
        for marker in assistant_markers + [
            "<|system|>",
            "system:",
            "<|user|>",
            "user:",
        ]:
            if (
                marker in normalized_output[: len(prompt) // 2]
            ):  # Check only the start for prompt markers
                is_likely_answer_only = False
                break

        if is_likely_answer_only:
            logger.warning(
                "Output does not seem to contain standard prompt markers, assuming entire output is the response."
            )
            response = normalized_output
            # Still try to remove end markers
            earliest_end_marker_pos = len(response)
            found_end_marker = None
            for end_marker in end_markers:
                pos = response.find(end_marker)
                if pos != -1 and pos < earliest_end_marker_pos:
                    earliest_end_marker_pos = pos
                    found_end_marker = end_marker

            if found_end_marker:
                logger.info(
                    f"DEBUG: Fallback 2 - Found end marker '{found_end_marker}' at position {earliest_end_marker_pos}"
                )
                response = response[:earliest_end_marker_pos].strip()

            if response and response.lower() != self.tokenizer.eos_token.lower():
                response = response.lstrip(":\\n ")  # Clean potential leading chars
                logger.info(
                    f"DEBUG: Fallback 2 final response (first 50 chars): '{response[:50]}'...\"\nDEBUG: (last 50 chars): ...'{response[-50:]}'"
                )
                # >>> ADDED DEBUG LOG: Exiting _extract_answer_from_output (Fallback 2)
                logger.info(
                    f"DEBUG_GARBLE: Exiting _extract_answer_from_output (Fallback 2) - First 100: '{response[:100]}'"
                )
                logger.info(
                    f"DEBUG_GARBLE: Exiting _extract_answer_from_output (Fallback 2) - Last 100: '{response[-100:]}'"
                )
                return response

        # If all methods fail, log detailed debugging info:
        logger.error(
            "Could not reliably extract assistant response from output using markers or prompt stripping."
        )
        logger.error(f"Prompt used (first 200 chars): {prompt[:200]}...")
        logger.error(
            f"Full model output (first 300 / last 300 chars):\nSTART>>>\n{normalized_output[:300]}\n...\n<<<END\n{normalized_output[-300:]}"
        )

        # As absolute last resort, return the original output if it's not too short,
        # hoping _clean_model_output can salvage something.
        if len(normalized_output) > 20:
            logger.warning(
                "Returning entire normalized output as last resort for cleaning."
            )
            # >>> ADDED DEBUG LOG: Exiting _extract_answer_from_output (Last Resort)
            logger.info(
                f"DEBUG_GARBLE: Exiting _extract_answer_from_output (Last Resort) - First 100: '{normalized_output[:100]}'"
            )
            logger.info(
                f"DEBUG_GARBLE: Exiting _extract_answer_from_output (Last Resort) - Last 100: '{normalized_output[-100:]}'"
            )
            return normalized_output

        # >>> ADDED DEBUG LOG: Exiting _extract_answer_from_output (Error Case)
        final_error_response = "I apologize, but I encountered an issue generating a proper response. Please try rephrasing your question."
        logger.info(
            f"DEBUG_GARBLE: Exiting _extract_answer_from_output (Error Case) - Returning: {final_error_response}"
        )
        return final_error_response

    def _prepare_context_metadata(
        self, context_docs: Optional[List[Document]]
    ) -> List[Dict[str, Any]]:
        """
        Prepare metadata from context documents for the response.

        Args:
            context_docs: List of context documents.

        Returns:
            List of dictionaries containing metadata for each document.
        """
        if not context_docs:
            return [
                {
                    "source": "No Context",
                    "content": "No relevant documents were used for this response.",
                    "file_name": "N/A",
                }
            ]

        metadata_list = []
        for doc in context_docs:
            file_name = doc.meta.get("file_name", "Unknown Source")
            content_preview = (
                doc.content[:250] + "..."
                if doc.content
                else "No content preview available."
            )
            metadata = {
                "source": f"Document (from {file_name})",
                "content": content_preview,
                "file_name": file_name,
                "score": (
                    doc.score if hasattr(doc, "score") else None
                ),  # Include score if available
                "meta": doc.meta,  # Include full meta for potential debugging
            }
            metadata_list.append(metadata)

        return metadata_list
