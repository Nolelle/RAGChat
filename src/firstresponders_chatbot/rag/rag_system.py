"""
RAG system module for the FirstRespondersChatbot project.

This module implements the backend logic for the Retrieval-Augmented Generation (RAG)
system, indexing uploaded PDF/text files, retrieving relevant context, and generating
responses using the fine-tuned Flan-T5-Small model.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import uuid

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

from haystack import Pipeline, Document
from haystack.components.converters import TextFileToDocument, PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
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
        model_dir: str = "tinyllama-1.1b-first-responder-fast",
        uploads_dir: str = "uploads",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 8,
        rerank_top_k: int = 5,
        use_hybrid_retrieval: bool = True,
    ):
        """
        Initialize the RAG system.

        Args:
            model_dir: Directory containing the fine-tuned model
            uploads_dir: Directory to store uploaded files
            embedding_model: Name of the embedding model to use
            top_k: Number of documents to retrieve
            rerank_top_k: Number of documents to keep after reranking
            use_hybrid_retrieval: Whether to use hybrid retrieval
        """
        # Set parameters
        self.model_dir = Path(model_dir)
        self.uploads_dir = Path(uploads_dir)
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.use_hybrid_retrieval = use_hybrid_retrieval

        # Create uploads directory if it doesn't exist
        os.makedirs(self.uploads_dir, exist_ok=True)

        # Initialize document store
        self.document_store = InMemoryDocumentStore()

        # Initialize embedders
        self.document_embedder = SentenceTransformersDocumentEmbedder(
            model=self.embedding_model
        )
        self.text_embedder = SentenceTransformersTextEmbedder(
            model=self.embedding_model
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

        # Load model and tokenizer
        self.model, self.tokenizer, self.device = self._load_model()

        # Track indexed files
        self.indexed_files = set()

        # Track sessions and their indexed files
        self.session_files = {}

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

    def _load_model(self):
        """
        Load the fine-tuned model and tokenizer.
        If the fine-tuned model doesn't exist, fall back to the base model.

        Returns:
            tuple: (model, tokenizer, device)
        """
        # Detect hardware
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Silicon acceleration")
            # Disable quantization for MPS (Apple Silicon)
            use_quantization = False
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
            use_quantization = True
        else:
            device = torch.device("cpu")
            logger.info("No GPU detected, using CPU (this might be slower)")
            use_quantization = False

        # Configure quantization for efficiency (only when not on MPS)
        if use_quantization:
            logger.info("Using 4-bit quantization for efficiency")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            logger.info("Quantization disabled - using standard precision")
            quantization_config = None

        # Try to load fine-tuned model if it exists
        if self.model_dir.exists():
            try:
                logger.info(
                    f"Attempting to load fine-tuned model from {self.model_dir}"
                )

                # Check if it's a TinyLlama model
                if "tinyllama" in str(self.model_dir).lower():
                    logger.info("Loading TinyLlama model")
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    # Load model with or without quantization based on device
                    if use_quantization:
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_dir,
                            quantization_config=quantization_config,
                            device_map="auto",
                            torch_dtype=torch.float16,
                        )
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_dir,
                            device_map="auto" if torch.cuda.is_available() else None,
                            torch_dtype=torch.float16,
                        )
                        # For MPS, manually move model to device
                        if device.type == "mps":
                            model = model.to(device)

                    tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

                    # Set padding token if not set
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token

                    logger.info("Successfully loaded fine-tuned TinyLlama model")
                    return model, tokenizer, device

                # Otherwise, try to load as a Phi-3 model
                from transformers import AutoModelForCausalLM

                # Load model with or without quantization based on device
                if use_quantization:
                    model = AutoModelForCausalLM.from_pretrained(
                        "microsoft/Phi-3-mini-4k-instruct",
                        quantization_config=quantization_config,
                        device_map="auto",
                        torch_dtype=torch.float16,
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        "microsoft/Phi-3-mini-4k-instruct",
                        device_map="auto" if torch.cuda.is_available() else None,
                        torch_dtype=torch.float16,
                    )
                    # For MPS, manually move model to device
                    if device.type == "mps":
                        model = model.to(device)

                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    "microsoft/Phi-3-mini-4k-instruct"
                )

                # Set padding token if not set
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # Try to load the LoRA adapter weights
                adapter_path = os.path.join(self.model_dir, "adapter")
                if os.path.exists(adapter_path):
                    logger.info(f"Loading LoRA adapter from {adapter_path}")
                    from peft import PeftModel

                    model = PeftModel.from_pretrained(model, adapter_path)
                else:
                    # If adapter doesn't exist, may need to try loading the full model
                    logger.info("LoRA adapter not found, trying to load full model")
                    if use_quantization:
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_dir,
                            quantization_config=quantization_config,
                            device_map="auto",
                            torch_dtype=torch.float16,
                        )
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_dir,
                            device_map="auto" if torch.cuda.is_available() else None,
                            torch_dtype=torch.float16,
                        )
                        # For MPS, manually move model to device
                        if device.type == "mps":
                            model = model.to(device)

                logger.info("Successfully loaded fine-tuned model")
            except Exception as e:
                logger.warning(f"Could not load fine-tuned model: {str(e)}")
                logger.info("Falling back to TinyLlama base model")
                from transformers import AutoModelForCausalLM, AutoTokenizer

                # Fall back to TinyLlama base model
                if use_quantization:
                    model = AutoModelForCausalLM.from_pretrained(
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        quantization_config=quantization_config,
                        device_map="auto",
                        torch_dtype=torch.float16,
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        device_map="auto" if torch.cuda.is_available() else None,
                        torch_dtype=torch.float16,
                    )
                    # For MPS, manually move model to device
                    if device.type == "mps":
                        model = model.to(device)

                tokenizer = AutoTokenizer.from_pretrained(
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
        else:
            # Use TinyLlama base model as fallback
            logger.info("Fine-tuned model not found, using TinyLlama base model")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if use_quantization:
                model = AutoModelForCausalLM.from_pretrained(
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=torch.float16,
                )
                # For MPS, manually move model to device
                if device.type == "mps":
                    model = model.to(device)

            tokenizer = AutoTokenizer.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer, device

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
                splitter = DocumentSplitter(
                    split_by="word",  # Split by word for better context preservation
                    split_length=150,  # Larger chunks to maintain context
                    split_overlap=30,  # Decent overlap to prevent context loss
                )
            else:
                splitter = DocumentSplitter(
                    split_by="sentence", split_length=3, split_overlap=1
                )

            # Create a pipeline
            pipeline = Pipeline()
            pipeline.add_component("splitter", splitter)

            # Add appropriate converter based on file type
            if file_path_obj.suffix.lower() == ".pdf":
                logger.info("PDF file detected - using PyPDFToDocument converter")
                pipeline.add_component("converter", pdf_converter)
                # Log PDF-specific details
                logger.info(f"PDF file size: {os.path.getsize(file_path)} bytes")
            elif file_path_obj.suffix.lower() in [".txt", ".md"]:
                pipeline.add_component("converter", text_converter)
            else:
                logger.warning(f"Unsupported file type: {file_path}")
                return []

            # Connect converter to splitter
            pipeline.connect("converter.documents", "splitter.documents")

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

            # Process file
            documents = self.process_file(file_path)
            if not documents:
                logger.error("No documents were created during processing")
                return False

            logger.info(f"Created {len(documents)} documents from file")
            logger.info(f"Sample document content: {documents[0].content[:200]}")

            # Add session ID to document metadata
            for doc in documents:
                doc.meta["session_id"] = session_id
                doc.meta["file_name"] = os.path.basename(file_path)
                doc.meta["file_path"] = str(file_path)

            # Log document details before embedding
            logger.info("Document details before embedding:")
            for i, doc in enumerate(documents[:3]):
                logger.info(f"Document {i} content preview: {doc.content[:200]}...")
                logger.info(f"Document {i} metadata: {doc.meta}")

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

            logger.info(
                f"Successfully indexed file: {file_path} for session: {session_id}"
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

            # Use a simplified approach that works for all queries
            # Focus on semantic similarity without special cases
            try:
                logger.info("Performing semantic retrieval for query...")
                embedded_query = self.text_embedder.run(text=query)
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
                bm25_result = self.bm25_retriever.run(query=query, top_k=self.top_k * 2)
                bm25_docs = bm25_result["documents"]
                logger.info(f"BM25 retriever returned {len(bm25_docs)} documents")

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

            # Retrieve context if not provided
            if context_docs is None:
                context_docs = self.retrieve_context(query, session_id=session_id)
                logger.info(
                    f"Retrieved {len(context_docs)} context documents for session {session_id}"
                )

            if not context_docs:
                logger.warning("No context documents retrieved")
                # Try to generate a response using the model's pre-trained knowledge
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

            # Log the context string length
            logger.info(f"Context string length: {len(context_str)} chars")

            # Generate prompt based on model type
            prompt = self._create_prompt_with_context(query, context_str)

            # Truncate prompt if necessary
            if len(prompt) > 8000:  # Conservative limit for most models
                logger.warning(f"Prompt too long ({len(prompt)} chars), truncating")
                prompt = self._truncate_prompt(prompt, 8000)

            # Log the prompt
            logger.info(f"PROMPT (first 500 chars):\n{prompt[:500]}...")

            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate the answer with parameters appropriate for the model
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=350,  # Increased for comprehensive summaries
                    min_new_tokens=75,  # Ensure substantive responses
                    temperature=0.7,  # Slightly increased for creative summarization
                    top_p=0.9,
                    top_k=40,
                    repetition_penalty=1.2,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode the output
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the assistant's response based on model type
            answer = self._extract_answer_from_output(full_output, prompt)

            # Get context metadata for response
            context_metadata = self._prepare_context_metadata(context_docs)

            # Create response
            response = {
                "query": query,
                "answer": answer,
                "context": context_metadata,
            }

            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "query": query,
                "answer": "Sorry, I encountered an error while processing your request.",
                "context": [],
            }

    def _format_context_for_prompt(self, context_docs: List[Document]) -> str:
        """Format the context documents into a string for the prompt."""
        context_str = ""
        for i, doc in enumerate(context_docs):
            # Include document number and file source in the context
            file_name = doc.meta.get("file_name", "Unknown source")
            context_str += f"Document {i+1} (from {file_name}):\n{doc.content}\n\n"

        # Trim if too long
        if len(context_str) > 6000:
            logger.warning(
                f"Context too long ({len(context_str)} chars), trimming to 6000 chars"
            )
            context_str = context_str[:6000] + "..."

        return context_str

    def _create_prompt_with_context(self, query: str, context_str: str) -> str:
        """Create a prompt with the context for the given model type."""
        # Check if we're using TinyLlama model
        if "TinyLlama" in self.tokenizer.name_or_path:
            logger.info("Using TinyLlama prompt format")
            # Format prompt for TinyLlama with instructions to summarize
            prompt = f"""<s>[INST] <<SYS>>
You are a first responder assistant designed to provide accurate, concise information based on official protocols and emergency response manuals. 
Answer questions using the provided context information, but DO NOT copy the information verbatim. 
Instead, synthesize and summarize the key points into a well-organized response that addresses the query.
Focus on delivering complete, accurate responses that address the core purpose and function of the equipment or procedures being discussed.
Use your own words to explain concepts clearly while maintaining factual accuracy.

When appropriate, use markdown formatting to enhance readability:
- Use bullet points or numbered lists for sequential steps, multiple items, or procedures
- Use simple tables for comparing multiple items with similar properties
- Use headers with # or ## to separate major sections if the response is lengthy
- Use **bold** or *italic* for emphasis when needed

For example, format a list like this:
1. First item
2. Second item
3. Third item

Format bullets like this:
- First point
- Second point
- Third point

Format a table like this:
| Item | Description |
|------|-------------|
| Item1 | Description1 |
| Item2 | Description2 |
<</SYS>>

I need information about the following topic: {query}

Here's the relevant information from my knowledge base:
{context_str} [/INST]"""
        else:
            logger.info("Using Phi-3 prompt format")
            # Format prompt for Phi-3 with instructions to summarize
            prompt = f"""<|system|>
You are a first responder assistant designed to provide accurate, concise information based on official protocols and emergency response manuals.
Answer questions using the provided context information, but DO NOT copy the information verbatim.
Instead, synthesize and summarize the key points into a well-organized response that addresses the query.
Focus on delivering complete, accurate responses that address the core purpose and function of the equipment or procedures being discussed.
Use your own words to explain concepts clearly while maintaining factual accuracy.

When appropriate, use markdown formatting to enhance readability:
- Use bullet points or numbered lists for sequential steps, multiple items, or procedures
- Use simple tables for comparing multiple items with similar properties
- Use headers with # or ## to separate major sections if the response is lengthy
- Use **bold** or *italic* for emphasis when needed

For example, format a list like this:
1. First item
2. Second item
3. Third item

Format bullets like this:
- First point
- Second point
- Third point

Format a table like this:
| Item | Description |
|------|-------------|
| Item1 | Description1 |
| Item2 | Description2 |
<|user|>
I need information about the following topic: {query}

Here's the relevant information from my knowledge base:
{context_str}
<|assistant|>"""

        return prompt

    def _truncate_prompt(self, prompt: str, max_length: int) -> str:
        """Truncate the prompt to the given maximum length while preserving structure."""
        if len(prompt) <= max_length:
            return prompt

        # For TinyLlama
        if "TinyLlama" in self.tokenizer.name_or_path:
            # Split into parts: system, query, context, and end tag
            parts = prompt.split(
                "Here's the relevant information from my knowledge base:"
            )
            if len(parts) == 2:
                prefix = (
                    parts[0] + "Here's the relevant information from my knowledge base:"
                )
                context_and_end = parts[1]

                # Split context and end tag
                if " [/INST]" in context_and_end:
                    context, end = context_and_end.split(" [/INST]", 1)
                    # Calculate how much we need to truncate
                    available_space = max_length - len(prefix) - len(end) - 3  # Buffer
                    if available_space > 100:  # Ensure we have reasonable space
                        truncated_context = context[:available_space] + "..."
                        return prefix + truncated_context + " [/INST]"

        # For Phi-3
        else:
            # Split into system and user parts
            if "<|user|>" in prompt and "<|assistant|>" in prompt:
                system_part = prompt.split("<|user|>")[0]
                user_part = (
                    "<|user|>" + prompt.split("<|user|>")[1].split("<|assistant|>")[0]
                )
                assistant_part = "<|assistant|>"

                # Calculate available space
                available_space = (
                    max_length - len(system_part) - len(assistant_part) - 3
                )
                if available_space > 100:
                    # Find where the context starts
                    if (
                        "Here's the relevant information from my knowledge base:"
                        in user_part
                    ):
                        prefix, context = user_part.split(
                            "Here's the relevant information from my knowledge base:", 1
                        )
                        prefix += (
                            "Here's the relevant information from my knowledge base:"
                        )
                        # Truncate context
                        available_space = (
                            max_length
                            - len(system_part)
                            - len(prefix)
                            - len(assistant_part)
                            - 3
                        )
                        truncated_context = context[:available_space] + "..."
                        return system_part + prefix + truncated_context + assistant_part

        # Default fallback: simple truncation
        return prompt[: max_length - 3] + "..."

    def _extract_answer_from_output(self, full_output: str, prompt: str) -> str:
        """Extract the model's answer from the full output."""
        # For TinyLlama
        if "TinyLlama" in self.tokenizer.name_or_path:
            # For TinyLlama, the answer is everything after the prompt
            answer = full_output[len(prompt.replace("[/INST]", "")) :]
            # Clean up any potential context prefixes
            answer = answer.replace("Answer Context:", "").strip()
        else:
            # For Phi-3, extract after the last <|assistant|> token
            response_parts = full_output.split("<|assistant|>")
            answer = response_parts[-1].strip()
            # Clean up any context prefixes
            answer = answer.replace("Answer Context:", "").strip()

        return answer

    def _prepare_context_metadata(self, context_docs: List[Document]) -> List[Dict]:
        """Prepare context metadata for the response."""
        context_metadata = []
        for doc in context_docs:
            meta = {
                "file_name": doc.meta.get("file_name", "Unknown"),
                "file_path": doc.meta.get("file_path", "Unknown"),
                "snippet": (
                    doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                ),
            }
            context_metadata.append(meta)
        return context_metadata

    def save_uploaded_file(
        self, file_data, filename: str, session_id: str = "default"
    ) -> str:
        """
        Save an uploaded file to the uploads directory.

        Args:
            file_data: The file data
            filename: The original filename
            session_id: Session ID to associate with this file

        Returns:
            str: Path to the saved file
        """
        # Generate a unique filename to avoid collisions
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = self.uploads_dir / unique_filename

        # Save file
        with open(file_path, "wb") as f:
            f.write(file_data)

        logger.info(f"Saved uploaded file to {file_path} for session {session_id}")

        # Index the file for this session
        self.index_file(str(file_path), session_id=session_id)

        return str(file_path)

    def clear_index(self) -> None:
        """Clear the entire document index."""
        self.document_store.delete_documents()
        self.indexed_files.clear()
        self.session_files.clear()
        logger.info("Document index cleared")

    def generate_from_knowledge(self, query: str) -> Dict[str, Any]:
        """
        Generate a response using only the model's pre-trained knowledge when no context is available.
        """
        logger.info(f"Generating response from model knowledge for query: {query}")

        try:
            # Check if we're using TinyLlama model
            if "TinyLlama" in self.tokenizer.name_or_path:
                # Format prompt for TinyLlama
                prompt = f"""<s>[INST] <<SYS>>
You are a first responder assistant designed to provide accurate information based on your training. 
If you don't have enough information to answer accurately, acknowledge the limitations and provide 
general guidance where possible.
Answer in your own words with clear explanations rather than technical jargon whenever possible.
Organize your response with clear structure and use concise language.

When appropriate, use formatting to enhance readability:
- Use bullet points or numbered lists for sequential steps, multiple items, or procedures
- Use simple tables for comparing multiple items with similar properties
- Use headers to separate major sections if the response is lengthy

Example list format:
1. First item
2. Second item
3. Third item

Example bullet format:
- First point
- Second point
- Third point

Example simple table format:
| Item | Description |
|------|-------------|
| Item1 | Description1 |
| Item2 | Description2 |
<</SYS>>

{query} [/INST]"""
            else:
                # Format prompt for Phi-3
                prompt = f"""<|system|>
You are a first responder assistant designed to provide accurate information based on your training. 
If you don't have enough information to answer accurately, acknowledge the limitations and provide 
general guidance where possible.
Answer in your own words with clear explanations rather than technical jargon whenever possible.
Organize your response with clear structure and use concise language.

When appropriate, use formatting to enhance readability:
- Use bullet points or numbered lists for sequential steps, multiple items, or procedures
- Use simple tables for comparing multiple items with similar properties
- Use headers to separate major sections if the response is lengthy

Example list format:
1. First item
2. Second item
3. Third item

Example bullet format:
- First point
- Second point
- Third point

Example simple table format:
| Item | Description |
|------|-------------|
| Item1 | Description1 |
| Item2 | Description2 |
<|user|>
{query}
<|assistant|>"""

            # Log the prompt
            logger.info(f"KNOWLEDGE PROMPT:\n{prompt}")

            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate the answer
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=250,
                    min_new_tokens=30,
                    temperature=0.7,  # Slightly higher temperature for more diverse responses
                    top_p=0.9,
                    top_k=40,
                    repetition_penalty=1.2,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode the output
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the assistant's response based on model type
            if "TinyLlama" in self.tokenizer.name_or_path:
                answer = full_output[len(prompt.replace("[/INST]", "")) :].strip()
            else:
                response_parts = full_output.split("<|assistant|>")
                answer = response_parts[-1].strip()

            # Create a context message that matches the format expected by the frontend
            context_message = [
                {
                    "file_name": "Model Knowledge",
                    "file_path": "None",
                    "snippet": "The response was generated using the model's knowledge as no relevant documents were found.",
                }
            ]

            return {
                "query": query,
                "answer": answer,
                "context": context_message,
            }

        except Exception as e:
            logger.error(f"Error generating response from knowledge: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

            return {
                "query": query,
                "answer": "I apologize, but I don't have enough information to provide a specific answer to your question. Please try indexing relevant documents first.",
                "context": [],
            }

    def generate_response_with_docs(
        self, query: str, docs: List[Document]
    ) -> Dict[str, Any]:
        """
        Generate a response for a specific query using the provided document list.
        Used for direct document retrieval for targeted queries.

        Args:
            query: The user's query
            docs: The documents to use as context

        Returns:
            Dict containing the response and context information
        """
        try:
            # Create context string from documents
            context_str = ""
            for i, doc in enumerate(docs[:3]):  # Limit to top 3 docs
                context_str += f"Document {i+1}:\n{doc.content}\n\n"

            # Trim if too long
            if len(context_str) > 6000:
                logger.warning(
                    f"Context too long ({len(context_str)} chars), trimming to 6000 chars"
                )
                context_str = context_str[:6000] + "..."

            # Format prompt based on model type
            if "TinyLlama" in self.tokenizer.name_or_path:
                logger.info("Using TinyLlama prompt format")
                prompt = f"""<s>[INST] <<SYS>>
You are a first responder assistant designed to provide accurate, concise information based on official protocols and emergency response manuals. 
Answer questions using the provided context information, but DO NOT copy the information verbatim. 
Instead, synthesize and summarize the key points into a well-organized response that addresses the query.
Focus on delivering complete, accurate responses that address the core purpose and function of the equipment or procedures being discussed.
Use your own words to explain concepts clearly while maintaining factual accuracy.

When appropriate, use formatting to enhance readability:
- Use bullet points or numbered lists for sequential steps, multiple items, or procedures
- Use simple tables for comparing multiple items with similar properties
- Use headers to separate major sections if the response is lengthy

Example list format:
1. First item
2. Second item
3. Third item

Example bullet format:
- First point
- Second point
- Third point

Example simple table format:
| Item | Description |
|------|-------------|
| Item1 | Description1 |
| Item2 | Description2 |
<</SYS>>

I need information about the following topic: {query}

Here's the relevant information:
{context_str} [/INST]"""
            else:
                logger.info("Using Phi-3 prompt format")
                prompt = f"""<|system|>
You are a first responder assistant designed to provide accurate, concise information based on official protocols and emergency response manuals.
Answer questions using the provided context information, but DO NOT copy the information verbatim.
Instead, synthesize and summarize the key points into a well-organized response that addresses the query.
Focus on delivering complete, accurate responses that address the core purpose and function of the equipment or procedures being discussed.
Use your own words to explain concepts clearly while maintaining factual accuracy.

When appropriate, use formatting to enhance readability:
- Use bullet points or numbered lists for sequential steps, multiple items, or procedures
- Use simple tables for comparing multiple items with similar properties
- Use headers to separate major sections if the response is lengthy

Example list format:
1. First item
2. Second item
3. Third item

Example bullet format:
- First point
- Second point
- Third point

Example simple table format:
| Item | Description |
|------|-------------|
| Item1 | Description1 |
| Item2 | Description2 |
<|user|>
I need information about the following topic: {query}

Here's the relevant information:
{context_str}
<|assistant|>"""

            # Log the prompt
            logger.info(f"DIRECT SEARCH PROMPT:\n{prompt}")

            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate the answer
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=350,
                    min_new_tokens=75,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    repetition_penalty=1.2,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode the output
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract the answer based on model type
            if "TinyLlama" in self.tokenizer.name_or_path:
                answer = full_output[len(prompt.replace("[/INST]", "")) :].strip()
            else:
                response_parts = full_output.split("<|assistant|>")
                answer = response_parts[-1].strip()

            # Get context metadata for response
            context_metadata = []
            for doc in docs:
                meta = {
                    "file_name": doc.meta.get("file_name", "Unknown"),
                    "file_path": doc.meta.get("file_path", "Unknown"),
                    "snippet": (
                        doc.content[:200] + "..."
                        if len(doc.content) > 200
                        else doc.content
                    ),
                }
                context_metadata.append(meta)

            # Create response
            response = {
                "query": query,
                "answer": answer,
                "context": context_metadata,
            }

            return response

        except Exception as e:
            logger.error(f"Error generating response with direct docs: {str(e)}")
            logger.exception("Full traceback:")
            return {
                "query": query,
                "answer": "Sorry, I encountered an error while processing your request.",
                "context": [],
            }
