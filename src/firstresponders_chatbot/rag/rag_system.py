"""
RAG system module for the FirstRespondersChatbot project.

This module implements the backend logic for the Retrieval-Augmented Generation (RAG)
system, indexing uploaded PDF/text files, retrieving relevant context, and generating
responses using the fine-tuned Llama 2 model.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import uuid
from collections import defaultdict

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

        # For model output, do a quick check for reasonable text
        if is_model_output:
            # If it contains common words and reasonable sentence structure, it's likely fine
            import re

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

            # If it has several common words, it's probably not garbled
            if common_words_count >= 3:
                # Also check for proper punctuation and sentence structure
                sentences = re.split(r"[.!?]+", text)
                proper_sentences = sum(
                    1
                    for s in sentences
                    if len(s.strip()) > 10 and s.strip()[0].isupper()
                )

                # If we have proper sentences, consider it valid
                if proper_sentences >= 1:
                    return False

        # Check for high percentage of non-ASCII characters (potential encoding issues)
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        ascii_threshold = (
            0.3 if not is_model_output else 0.4
        )  # Less strict for model output
        if len(text) > 0 and non_ascii_count / len(text) > ascii_threshold:
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
        # Use different thresholds based on source
        entropy_threshold = (
            5.5 if not is_model_output else 6.0
        )  # Higher threshold for model output
        if entropy > entropy_threshold:
            return True

        # Check for unusual character sequences (random distribution of special chars)
        import re

        # Look for sequences like "iaIAs" or "síaà" that are unlikely in normal text
        unusual_pattern = r"([^\w\s]{3,}|([a-zA-Z][^a-zA-Z]){4,})"  # Reduced sequence length requirements
        if re.search(unusual_pattern, text):
            return True

        # Additional check for concatenated words without spaces (common in bad PDF extraction)
        words_without_spaces = re.findall(r"[a-zA-Z]{20,}", text)
        if (
            words_without_spaces
            and len("".join(words_without_spaces)) / len(text) > 0.2
        ):
            return True

        return False

    def process_file(self, file_path: str) -> List[Document]:
        """
        Process a file and split it into documents.

        Args:
            file_path: Path to the file to process

        Returns:
            List[Document]: List of processed documents
        """
        logger.info(f"Processing file: {file_path}")

        try:
            # Create pipeline components
            pdf_converter = PyPDFToDocument()
            text_converter = TextFileToDocument()

            # Use different splitting parameters for PDFs vs text files
            file_path_obj = Path(file_path)
            if file_path_obj.suffix.lower() == ".pdf":
                logger.info(
                    "PDF file detected - using PDF-optimized splitting parameters"
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
                    "PDF file detected - using PyPDFToDocument converter with cleaning"
                )

                # For PDFs, perform pre-check with PyPDF2 to assess potential issues
                try:
                    import PyPDF2

                    with open(file_path, "rb") as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        num_pages = len(pdf_reader.pages)
                        logger.info(f"PDF has {num_pages} pages")

                        # Try to extract text from first page to check for encoding issues
                        first_page_text = pdf_reader.pages[0].extract_text()
                        if self._detect_garbled_text(first_page_text):
                            logger.warning(
                                "PDF first page contains potential encoding issues"
                            )
                except Exception as e:
                    logger.warning(f"Could not perform PDF pre-check: {str(e)}")

                pipeline.add_component("converter", pdf_converter)
                pipeline.add_component("cleaner", cleaner)
                pipeline.add_component("splitter", splitter)

                # Connect components in the pipeline
                pipeline.connect("converter.documents", "cleaner.documents")
                pipeline.connect("cleaner.documents", "splitter.documents")

                # Log PDF-specific details
                logger.info(f"PDF file size: {os.path.getsize(file_path)} bytes")
            elif file_path_obj.suffix.lower() in [".txt", ".md"]:
                pipeline.add_component("converter", text_converter)
                pipeline.add_component("splitter", splitter)

                # Connect components in the pipeline
                pipeline.connect("converter.documents", "splitter.documents")
            else:
                logger.warning(f"Unsupported file type: {file_path}")
                return []

            # Run pipeline
            logger.info("Starting document conversion pipeline...")
            result = pipeline.run({"converter": {"sources": [file_path]}})

            if "converter" in result:
                logger.info(
                    f"Converter output: {len(result['converter'].get('documents', []))} documents"
                )
                # Log sample of converted content
                if result["converter"].get("documents"):
                    sample = result["converter"]["documents"][0].content[:500]
                    logger.info(f"Sample converted content: {sample}...")

                    # Early detection for severely corrupted PDFs
                    if (
                        file_path_obj.suffix.lower() == ".pdf"
                        and self._detect_garbled_text(sample)
                    ):
                        logger.warning(
                            "Initial PDF conversion appears to contain garbled text"
                        )

            # Check if splitter produced documents
            if "splitter" not in result or not result["splitter"].get("documents"):
                if file_path_obj.suffix.lower() == ".pdf":
                    logger.error(
                        "Splitter did not produce any documents. Attempting fallback PDF extraction using PyPDF2."
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
                                "Fallback extraction produced insufficient text."
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
                            f"Fallback extraction produced {len(fallback_docs)} documents."
                        )
                        result["splitter"] = {"documents": fallback_docs}
                    except Exception as e:
                        logger.error("Fallback extraction failed: " + str(e))
                        return []
                else:
                    logger.error(
                        "Splitter did not produce any documents for non-PDF file."
                    )
                    return []

            documents = result["splitter"]["documents"]

            # Log chunk details
            logger.info(f"Document splitting results:")
            logger.info(f"Number of chunks: {len(documents)}")
            if documents:
                avg_chunk_size = sum(len(doc.content) for doc in documents) / len(
                    documents
                )
                logger.info(f"Average chunk size: {avg_chunk_size:.2f} characters")
                logger.info(f"Sample chunk: {documents[0].content[:200]}...")

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
                    logger.warning(f"Detected likely garbled text in document chunk")

                # Clean content - enhanced cleaning procedure
                import re

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
                            f"Skipping garbled chunk that couldn't be cleaned"
                        )
                        continue

                # Create a new document with the cleaned content
                cleaned_doc = Document(content=content, meta=doc.meta)
                cleaned_doc.id = doc.id

                # Add to cleaned documents list
                cleaned_documents.append(cleaned_doc)

            # Log counts of cleaned and skipped chunks
            logger.info(f"Original chunk count: {len(documents)}")
            logger.info(f"Cleaned chunk count: {len(cleaned_documents)}")
            logger.info(f"Detected {garbled_chunks_count} garbled chunks")
            logger.info(
                f"Skipped {len(documents) - len(cleaned_documents)} chunks due to unrecoverable garbled text"
            )

            # Check if we have at least some usable documents
            if not cleaned_documents:
                if file_path_obj.suffix.lower() == ".pdf":
                    logger.warning(
                        "Cleaning removed all document content. Attempting fallback extraction using PyPDF2 for the PDF."
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
                                "Fallback extraction produced insufficient text."
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
                            logger.error("Fallback extraction produced no documents.")
                            return []
                        cleaned_documents = fallback_docs
                        logger.info(
                            f"Fallback extraction after cleaning produced {len(cleaned_documents)} documents."
                        )
                    except Exception as e:
                        logger.error(
                            "Fallback extraction after cleaning failed: " + str(e)
                        )
                        return []
                else:
                    logger.error(
                        "No usable documents could be extracted from the file after cleaning"
                    )
                    return []

            # If more than 80% of chunks were garbled, warn about potential issues
            if garbled_chunks_count / len(documents) > 0.8:
                logger.warning(
                    f"More than 80% of document chunks contained garbled text. Results may be unreliable."
                )

            # Replace original documents with cleaned ones
            documents = cleaned_documents

            # Add file metadata to documents
            for doc in documents:
                doc.meta["file_name"] = file_path_obj.name
                doc.meta["file_path"] = str(file_path)

            logger.info(f"Split {file_path_obj.name} into {len(documents)} chunks")
            return documents

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            logger.exception("Full traceback:")
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
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False

            # Initialize session if not exists
            if session_id not in self.session_files:
                self.session_files[session_id] = set()

            # For PDFs, perform a pre-check to verify readability
            file_path_obj = Path(file_path)
            if file_path_obj.suffix.lower() == ".pdf":
                try:
                    import PyPDF2

                    with open(file_path, "rb") as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        num_pages = len(pdf_reader.pages)
                        logger.info(f"PDF pre-check: {file_path} has {num_pages} pages")

                        # Attempt to extract text from the first page
                        first_page_text = pdf_reader.pages[0].extract_text()
                        if not first_page_text or len(first_page_text.strip()) < 10:
                            logger.warning(
                                f"PDF pre-check: {file_path} contains very little text in first page"
                            )
                except Exception as e:
                    logger.warning(f"PDF pre-check failed for {file_path}: {str(e)}")

            # Process file
            documents = self.process_file(file_path)
            if not documents:
                logger.error("No documents were created during processing")
                return False

            logger.info(f"Created {len(documents)} documents from file")
            logger.info(f"Sample document content: {documents[0].content[:200]}")

            # Double-check document quality
            valid_docs = 0
            for doc in documents:
                if (
                    doc.content
                    and len(doc.content.strip()) > 20
                    and not self._detect_garbled_text(doc.content)
                ):
                    valid_docs += 1

            if valid_docs < len(documents) * 0.3:  # Less than 30% valid
                logger.warning(
                    f"Only {valid_docs} out of {len(documents)} documents appear to be valid quality"
                )
                if valid_docs == 0:
                    logger.error("No valid documents found, aborting indexing")
                    return False

            # Add session ID to document metadata
            for doc in documents:
                doc.meta["session_id"] = session_id
                doc.meta["file_name"] = os.path.basename(file_path)
                doc.meta["file_path"] = str(file_path)

            # Log document details before embedding
            logger.info("Document details before embedding:")
            for i, doc in enumerate(documents[:3]):
                logger.info(f"Document {i+1} content preview: {doc.content[:200]}...")
                logger.info(f"Document {i+1} metadata: {doc.meta}")

            # Embed documents
            logger.info("Starting document embedding...")
            embedded_documents = self.document_embedder.run(documents=documents)[
                "documents"
            ]
            logger.info(f"Embedded {len(embedded_documents)} documents")

            # Verify embeddings
            missing_embeddings = 0
            for i, doc in enumerate(embedded_documents):
                if not hasattr(doc, "embedding") or doc.embedding is None:
                    missing_embeddings += 1
                    logger.error(f"Document {i} is missing embeddings!")
                elif i < 3:  # Log first 3 docs
                    logger.info(f"Document {i} embedding shape: {len(doc.embedding)}")
                    logger.info(
                        f"Document {i} embedding sample: {doc.embedding[:5]}..."
                    )  # Log first 5 values

            if missing_embeddings > 0:
                logger.error(f"{missing_embeddings} documents are missing embeddings!")
                return False

            # Write to document store
            initial_count = self.document_store.count_documents()
            logger.info(f"Document store count before writing: {initial_count}")

            # Write new documents
            self.document_store.write_documents(embedded_documents)
            final_count = self.document_store.count_documents()
            logger.info(f"Document store count after writing: {final_count}")

            documents_added = final_count - initial_count
            logger.info(f"Added {documents_added} new documents to store")

            # Verify documents were added
            if documents_added <= 0:
                logger.error("No new documents were added to the store")
                return False

            # Mark file as indexed for this session
            self.indexed_files.add(file_path)
            self.session_files[session_id].add(file_path)

            # Verify indexed documents for this session and file
            session_file_docs = 0
            for doc in self.document_store.filter_documents():
                if doc.meta.get("session_id") == session_id and str(
                    doc.meta.get("file_path")
                ) == str(file_path):
                    session_file_docs += 1

            if session_file_docs == 0:
                logger.error(
                    f"Verification failed: No documents found for {file_path} in session {session_id}"
                )
                return False

            logger.info(
                f"Successfully indexed file: {file_path} for session: {session_id} with {session_file_docs} documents"
            )
            return True

        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {str(e)}")
            logger.exception("Full traceback:")
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
            import re

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

            # Decode the output
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the assistant's response
            answer = self._extract_answer_from_output(full_output, prompt)

            # Check if the answer appears garbled
            if self._detect_garbled_text(answer, is_model_output=True):
                logger.warning(
                    "Generated answer appears to be garbled, generating fallback response"
                )
                # Create a simpler prompt and try again
                simpler_prompt = f"<|system|>\nAnswer this question briefly and clearly: {query}\n<|user|>\n{query}\n<|assistant|>"
                simpler_inputs = self.tokenizer(simpler_prompt, return_tensors="pt").to(
                    self.device
                )

                with torch.no_grad():
                    simpler_outputs = self.model.generate(
                        **simpler_inputs,
                        max_new_tokens=300,
                        temperature=0.4,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                simpler_output = self.tokenizer.decode(
                    simpler_outputs[0], skip_special_tokens=True
                )
                answer = self._extract_answer_from_output(
                    simpler_output, simpler_prompt
                )

                # If still garbled, use a generic response
                if self._detect_garbled_text(answer, is_model_output=True):
                    answer = "I apologize, but I'm having trouble generating a clear response. Please try reformulating your question."

            # Provide detailed context information and ensure proper source attribution
            context_metadata = self._prepare_context_metadata(context_docs)

            # Add "Generated by Llama 2 Model" to each answer for attribution
            if not answer.endswith("[Generated by Llama 2 Model]"):
                answer += "\n\n[Generated by Llama 2 Model]"

            return {
                "query": query,
                "answer": answer,
                "context": context_metadata,
            }

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            logger.exception("Full traceback:")
            return {
                "query": query,
                "answer": "I apologize, but I'm having trouble processing your request. Please try again or rephrase your question.",
                "context": [
                    {
                        "source": "Error (no documents available)",
                        "content": "An error occurred while processing your request.",
                        "file_name": "Error",
                    }
                ],
            }

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
                import re

                content = re.sub(r"[^\x20-\x7E]", " ", content)
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
        """Create a properly formatted prompt for Phi-4 with context."""
        # Phi-4 prompt format adapted for RAG
        context_section = (
            f"<|context|>{context_str}<|endofcontext|>\n\n" if context_str else ""
        )
        return (
            f"<|system|>\n"
            f"You are a knowledgeable first responder assistant. Use the provided context to answer the question accurately. "
            f"If the context doesn't contain the answer, state that clearly.\n"
            f"Guidelines:\n"
            f"1. Base answers strictly on the provided context.\n"
            f"2. Be concise and precise.\n"
            f"3. If unsure or the answer isn't in the context, say so.\n"
            f"4. Prioritize safety information.\n"
            f"{context_section}"
            f"<|user|>\n{query}\n<|assistant|>"
        )
