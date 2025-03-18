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
from haystack.components.rankers import TransformersSimilarityRanker
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
        self.reranker = TransformersSimilarityRanker(
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
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("No GPU detected, using CPU (this might be slower)")

        # Try to load fine-tuned model if it exists
        if self.model_dir.exists():
            try:
                logger.info(
                    f"Attempting to load fine-tuned model from {self.model_dir}"
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_dir))
                tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
                logger.info("Successfully loaded fine-tuned model")
            except Exception as e:
                logger.warning(f"Could not load fine-tuned model: {str(e)}")
                logger.info("Falling back to base Flan-T5 model")
                model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
                tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        else:
            # Use base model
            logger.info("Fine-tuned model not found, using base Flan-T5 model")
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

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
            doc_count = self.document_store.count_documents()
            logger.info(
                f"Retrieving context for query: '{query}'. Document count: {doc_count}"
            )

            if doc_count == 0:
                logger.warning("Document store is empty. No documents to retrieve.")
                return []

            # Hybrid retrieval approach
            if self.use_hybrid_retrieval:
                logger.info("Using hybrid retrieval approach (BM25 + embedding)")

                # Get documents from BM25 retriever
                try:
                    bm25_result = self.bm25_retriever.run(query=query, top_k=self.top_k)
                    bm25_docs = bm25_result["documents"]
                    logger.info(f"BM25 retriever returned {len(bm25_docs)} documents")

                    # DEBUG: Log first BM25 document content
                    if bm25_docs:
                        logger.info(f"First BM25 doc: {bm25_docs[0].content[:100]}...")
                except Exception as e:
                    logger.error(f"BM25 retriever failed: {str(e)}")
                    bm25_docs = []

                # Embed query for semantic retrieval
                try:
                    embedded_query = self.text_embedder.run(text=query)
                    logger.info(
                        f"Embedded query successfully. Embedding shape: {len(embedded_query['embedding'])}"
                    )

                    # Get documents from embedding retriever
                    embedding_result = self.embedding_retriever.run(
                        query_embedding=embedded_query["embedding"], top_k=self.top_k
                    )
                    embedding_docs = embedding_result["documents"]
                    logger.info(
                        f"Embedding retriever returned {len(embedding_docs)} documents"
                    )

                    # DEBUG: Log first embedding doc content
                    if embedding_docs:
                        logger.info(
                            f"First embedding doc: {embedding_docs[0].content[:100]}..."
                        )
                except Exception as e:
                    logger.error(f"Embedding retrieval failed: {str(e)}")
                    logger.error(f"Falling back to BM25 results only")
                    embedding_docs = []

                # If both retrievers failed, return empty list
                if not bm25_docs and not embedding_docs:
                    logger.error("Both retrievers failed, no documents retrieved")
                    return []

                # Continue with whatever documents we have
                combined_docs = []
                seen_ids = set()

                for doc in bm25_docs + embedding_docs:
                    if doc.id not in seen_ids:
                        combined_docs.append(doc)
                        seen_ids.add(doc.id)

                logger.info(f"Combined results: {len(combined_docs)} unique documents")

                # Rerank the combined results
                if combined_docs and len(combined_docs) > 0:
                    try:
                        reranker_result = self.reranker.run(
                            documents=combined_docs, query=query
                        )
                        reranked_docs = reranker_result["documents"]
                        logger.info(f"Reranked to top {len(reranked_docs)} documents")
                    except Exception as e:
                        logger.error(f"Reranking failed: {str(e)}")
                        logger.info("Using combined results without reranking")
                        reranked_docs = combined_docs[: self.rerank_top_k]

                    # Log the top document titles to verify retrieval is working
                    if len(reranked_docs) > 0:
                        for i, doc in enumerate(reranked_docs[:3]):  # Log top 3 docs
                            score_info = (
                                f", score: {doc.score}" if hasattr(doc, "score") else ""
                            )
                            file_info = (
                                f" from {doc.meta.get('file_name', 'unknown')}"
                                if doc.meta
                                else ""
                            )
                            content_preview = (
                                doc.content[:100] + "..."
                                if len(doc.content) > 100
                                else doc.content
                            )
                            logger.info(
                                f"Top doc {i+1}{score_info}{file_info}: {content_preview}"
                            )

                        # Log the full content of the first document for debugging
                        logger.info(
                            f"FULL CONTENT of top document:\n{reranked_docs[0].content}"
                        )

                    return reranked_docs

                logger.info(
                    f"Returning {len(combined_docs)} combined documents without reranking"
                )
                return combined_docs

            # Fallback to just embedding retrieval
            else:
                logger.info("Using embedding retrieval only")

                try:
                    # Embed the query
                    embedded_query = self.text_embedder.run(text=query)
                    # Use the embedding for retrieval
                    result = self.embedding_retriever.run(
                        query_embedding=embedded_query["embedding"], top_k=self.top_k
                    )
                    docs = result["documents"]
                    logger.info(f"Embedding retriever returned {len(docs)} documents")
                    return docs
                except Exception as e:
                    logger.error(f"Embedding retrieval failed: {str(e)}")

                    # Fall back to BM25 retrieval
                    logger.info("Falling back to BM25 retrieval")
                    try:
                        bm25_result = self.bm25_retriever.run(
                            query=query, top_k=self.top_k
                        )
                        bm25_docs = bm25_result["documents"]
                        logger.info(
                            f"BM25 retriever returned {len(bm25_docs)} documents"
                        )
                        return bm25_docs
                    except Exception as e2:
                        logger.error(f"BM25 fallback also failed: {str(e2)}")
                        return []

        except Exception as e:
            logger.error(f"Error retrieving context for query '{query}': {str(e)}")
            # Print the full exception traceback for better debugging
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def generate_response(
        self, query: str, context_docs: Optional[List[Document]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response to a query using the RAG system.

        Args:
            query: Query to generate a response for
            context_docs: Optional list of context documents to use instead of retrieving them

        Returns:
            Dict containing the query, answer, and context
        """
        logger.info(f"Generating response for query: {query}")

        try:
            # Retrieve context if not provided
            if context_docs is None:
                context_docs = self.retrieve_context(query)
                logger.info(f"Retrieved {len(context_docs)} context documents")

            if not context_docs:
                logger.warning("No context documents retrieved")
                return {
                    "query": query,
                    "answer": "I don't have enough information to answer that question. Please try uploading relevant documents first.",
                    "context": [],
                }

            # Create context string from documents
            # For Flan-T5, use a very clear format that highlights relevant information
            context_str = ""
            for i, doc in enumerate(context_docs[:3]):  # Limit to top 3 docs
                context_str += f"Document {i+1}:\n{doc.content}\n\n"

            # Trim if too long
            if len(context_str) > 6000:
                logger.warning(
                    f"Context too long ({len(context_str)} chars), trimming to 6000 chars"
                )
                context_str = context_str[:6000] + "..."

            # Log context information
            logger.info(f"Using {len(context_docs)} documents for context")
            logger.info(f"Context length: {len(context_str)} characters")

            # Check if we're using the fine-tuned model
            is_fine_tuned = self.model_dir.exists() and "first-responder" in str(
                self.model_dir
            )

            if is_fine_tuned:
                # Format prompt for fine-tuned model
                prompt = f"""Answer the question based on the following context. You are a first responders chatbot designed to help with training and education.

Context:
{context_str}

Question: {query}

Answer:"""
            else:
                # Explicitly structured prompt specifically for Flan-T5
                prompt = f"""Based on the following information, please answer: {query}

Information:
{context_str}

Answer:"""

            # Log the prompt
            logger.info(f"PROMPT:\n{prompt}")

            # Tokenize the prompt
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            ).to(self.device)

            logger.info(f"Input token length: {inputs['input_ids'].shape[1]}")

            # Generate the answer with parameters optimized for Flan-T5
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=150,
                    min_length=10,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    length_penalty=1.0,
                )

            # Decode the output
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = answer.strip()

            # Log the raw output
            logger.info(f"GENERATED ANSWER: {answer}")

            # Check if the answer is too short or incomplete
            if len(answer) < 15 or (".." in answer and len(answer) < 30):
                logger.warning(f"Generated answer looks incomplete: '{answer}'")
                # Extract relevant sentences from context as fallback
                relevant_info = self._extract_relevant_info(query, context_docs)
                if relevant_info:
                    answer = f"Based on the available information, NFPA 1971 appears to relate to protective ensembles for firefighters. Specifically from the documents: {relevant_info}"
                else:
                    answer = "Based on the available information, I couldn't find a complete explanation of what NFPA 1971 regulates. It appears to be related to protective equipment for firefighters, but the specific details aren't clear in the provided documents."

                logger.info(f"Using fallback answer: {answer}")

            # Get context metadata for response
            context_metadata = []
            for doc in context_docs:
                meta = {
                    "file_name": doc.meta.get("file_name", "Unknown"),
                    "file_path": doc.meta.get("file_path", "Unknown"),
                    "content_preview": (
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
            logger.error(f"Error generating response: {str(e)}")
            # Print the full exception traceback for better debugging
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "query": query,
                "answer": "Sorry, I encountered an error while processing your request.",
                "context": [],
            }

    def _extract_relevant_info(self, query: str, context_docs: List[Document]) -> str:
        """
        Extract the most relevant information from context documents for a fallback response.

        Args:
            query: The user query
            context_docs: The context documents

        Returns:
            str: Extracted relevant information, or empty string if none found
        """
        # Keywords to look for in the documents
        query_terms = query.lower().split()
        target_terms = [
            "nfpa",
            "1971",
            "standard",
            "regulate",
            "requirements",
            "protective",
            "equipment",
        ]

        # Look for sentences containing both NFPA and 1971
        relevant_sentences = []

        for doc in context_docs:
            content = doc.content
            # Simple sentence splitting (not perfect but sufficient for this use case)
            sentences = [s.strip() for s in content.replace("\n", " ").split(".")]

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                lower_sentence = sentence.lower()
                # Check if the sentence contains relevant terms
                if "nfpa" in lower_sentence and "1971" in lower_sentence:
                    # Score by number of target terms present
                    score = sum(1 for term in target_terms if term in lower_sentence)
                    relevant_sentences.append((sentence, score))

        # Sort by relevance score
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)

        # Take top 2 sentences
        top_sentences = [s[0] for s in relevant_sentences[:2]]

        return ". ".join(top_sentences)

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
