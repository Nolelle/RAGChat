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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
from haystack.components.rankers import TransformersSentenceRanker
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG system for the FirstRespondersChatbot."""

    def __init__(
        self,
        model_dir: str = "flan-t5-large-first-responder",
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
        self.reranker = TransformersSentenceRanker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=self.rerank_top_k
        )

        # Load model and tokenizer
        self.model, self.tokenizer, self.device = self._load_model()

        # Track indexed files
        self.indexed_files = set()

        # Warm up the embedding models
        self.warm_up()

        logger.info("RAG system initialized successfully")

    def warm_up(self):
        """
        Warm up the embedding models by calling their warm_up() methods.
        This ensures the models are loaded before they are used.
        """
        try:
            logger.info("Warming up embedding models...")
            # Warm up document embedder
            self.document_embedder.warm_up()

            # Warm up text embedder
            self.text_embedder.warm_up()

            logger.info("Embedding models warmed up successfully")
        except Exception as e:
            logger.error(f"Error warming up embedding models: {str(e)}")
            raise

    def _load_model(self):
        """
        Load the fine-tuned model and tokenizer.

        Returns:
            tuple: (model, tokenizer, device)
        """
        # Check if model exists
        if not self.model_dir.exists():
            logger.error("Model directory not found. Please run train.py first.")
            raise FileNotFoundError(f"Model directory {self.model_dir} not found")

        # Detect hardware
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Silicon acceleration")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("No GPU detected, using CPU (this might be slower)")

        # Load model and tokenizer
        logger.info(f"Loading model from {self.model_dir}")
        model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

        # Move model to device
        model = model.to(device)

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
            splitter = DocumentSplitter(
                split_by="sentence", split_length=3, split_overlap=1
            )

            # Create a pipeline
            pipeline = Pipeline()
            pipeline.add_component("splitter", splitter)

            # Add appropriate converter based on file type
            file_path_obj = Path(file_path)
            if file_path_obj.suffix.lower() == ".pdf":
                pipeline.add_component("converter", pdf_converter)
            elif file_path_obj.suffix.lower() in [".txt", ".md"]:
                pipeline.add_component("converter", text_converter)
            else:
                logger.warning(f"Unsupported file type: {file_path}")
                return []

            # Connect converter to splitter
            pipeline.connect("converter.documents", "splitter.documents")

            # Run pipeline
            result = pipeline.run({"converter": {"sources": [file_path]}})
            documents = result["splitter"]["documents"]

            # Add file metadata to documents
            for doc in documents:
                doc.meta["file_name"] = file_path_obj.name
                doc.meta["file_path"] = str(file_path)

            logger.info(f"Split {file_path_obj.name} into {len(documents)} chunks")
            return documents

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []

    def index_file(self, file_path: str) -> bool:
        """
        Index a file for retrieval.

        Args:
            file_path: Path to the file to index

        Returns:
            bool: True if indexing was successful, False otherwise
        """
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False

        # Check if file is already indexed
        if file_path in self.indexed_files:
            logger.info(f"File already indexed: {file_path}")
            return True

        try:
            # Process file
            documents = self.process_file(file_path)
            if not documents:
                return False

            # Embed documents
            embedded_documents = self.document_embedder.run(documents=documents)[
                "documents"
            ]

            # Write to document store
            self.document_store.write_documents(embedded_documents)

            # Mark file as indexed
            self.indexed_files.add(file_path)

            logger.info(f"Successfully indexed file: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {str(e)}")
            return False

    def retrieve_context(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The user's query

        Returns:
            List[Document]: List of relevant documents
        """
        try:
            # Check if there are any documents in the store
            if self.document_store.count_documents() == 0:
                logger.warning("Document store is empty. No documents to retrieve.")
                return []

            # Hybrid retrieval approach
            if self.use_hybrid_retrieval:
                # Get documents from BM25 retriever
                bm25_docs = self.bm25_retriever.run(
                    {"query": query, "top_k": self.top_k}
                )["documents"]

                # Get documents from embedding retriever
                embedding_docs = self.embedding_retriever.run(
                    {"query": query, "top_k": self.top_k}
                )["documents"]

                # Combine results (removing duplicates)
                combined_docs = []
                seen_ids = set()

                for doc in bm25_docs + embedding_docs:
                    if doc.id not in seen_ids:
                        combined_docs.append(doc)
                        seen_ids.add(doc.id)

                # Rerank the combined results
                if combined_docs and len(combined_docs) > 0:
                    reranker_result = self.reranker.run(
                        {"documents": combined_docs, "query": query}
                    )
                    return reranker_result["documents"]
                return combined_docs

            # Fallback to just embedding retrieval
            else:
                return self.embedding_retriever.run(
                    {"query": query, "top_k": self.top_k}
                )["documents"]

        except Exception as e:
            logger.error(f"Error retrieving context for query '{query}': {str(e)}")
            return []

    def generate_response(
        self, query: str, context_docs: Optional[List[Document]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response to a query using the RAG approach.

        Args:
            query: The user's query
            context_docs: Optional list of context documents (if None, will retrieve context)

        Returns:
            Dict[str, Any]: Response with answer and context
        """
        try:
            # Retrieve context if not provided
            if context_docs is None:
                context_docs = self.retrieve_context(query)

            # Sort documents by relevance score if available
            if context_docs and hasattr(context_docs[0], "score"):
                context_docs = sorted(
                    context_docs,
                    key=lambda doc: doc.score if hasattr(doc, "score") else 0,
                    reverse=True,
                )

            # Format context with document separators and improved structure
            context_text = ""
            context_sources = []

            for i, doc in enumerate(context_docs):
                # Include metadata in context if available
                meta_info = ""
                if doc.meta:
                    if "file_name" in doc.meta:
                        meta_info += f" [Source: {doc.meta['file_name']}]"
                    if "page_number" in doc.meta:
                        meta_info += f" [Page: {doc.meta['page_number']}]"

                # Add document with improved formatting
                context_text += (
                    f"\n### Document {i+1}{meta_info}:\n{doc.content.strip()}\n"
                )

                # Track sources for return value
                if doc.meta and "file_name" in doc.meta:
                    source = {
                        "file_name": doc.meta["file_name"],
                        "snippet": (
                            doc.content[:100] + "..."
                            if len(doc.content) > 100
                            else doc.content
                        ),
                    }
                    if "page_number" in doc.meta:
                        source["page"] = doc.meta["page_number"]
                    if source not in context_sources:
                        context_sources.append(source)

            # Expert instruction prompt for better paraphrasing and information synthesis
            input_text = f"""You are a first responder education assistant designed to provide accurate, clear information based on training materials.

Answer the question based on ONLY the following context. Your goal is to synthesize the information into a helpful, educational response.

Never directly copy text from the documents. Instead:
1. Analyze the key points from all relevant sections
2. Connect related information across different documents
3. Explain concepts in simple, clear terms for someone in training
4. Structure your response in a logical, easy-to-understand format
5. Be conversational but professional in your tone
6. If the context doesn't contain enough information, acknowledge limitations

Context:
{context_text}

Question: {query}

Answer:"""

            # Tokenize and move to device
            input_ids = self.tokenizer(
                input_text, return_tensors="pt", truncation=True, max_length=512
            ).input_ids.to(self.device)

            # Generate output with optimized parameters for better responses
            outputs = self.model.generate(
                input_ids,
                max_length=300,  # Increased for more comprehensive answers
                min_length=75,  # Ensure responses have reasonable length
                num_beams=5,
                temperature=0.8,  # Balanced for creativity and accuracy
                no_repeat_ngram_size=3,
                early_stopping=True,
                do_sample=True,
                top_k=50,  # Consider more tokens for each step
                top_p=0.92,
                repetition_penalty=1.2,  # Discourage repetition
                length_penalty=1.5,  # Encourage longer responses
            )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return {"answer": response, "context": context_sources, "query": query}

        except Exception as e:
            logger.error(f"Error generating response for query '{query}': {str(e)}")
            return {
                "answer": f"Error generating response: {str(e)}",
                "context": [],
                "query": query,
            }

    def save_uploaded_file(self, file_data, filename: str) -> str:
        """
        Save an uploaded file to the uploads directory.

        Args:
            file_data: The file data
            filename: The original filename

        Returns:
            str: Path to the saved file
        """
        # Generate a unique filename to avoid collisions
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = self.uploads_dir / unique_filename

        # Save file
        with open(file_path, "wb") as f:
            f.write(file_data)

        logger.info(f"Saved uploaded file to {file_path}")
        return str(file_path)

    def clear_index(self) -> None:
        """Clear the document index."""
        self.document_store.delete_documents()
        self.indexed_files.clear()
        logger.info("Document index cleared")
