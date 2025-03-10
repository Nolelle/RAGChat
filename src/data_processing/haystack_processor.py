from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
import os
from haystack import Document, Pipeline
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack.components.retrievers import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)

# Try to import FAISS if available
try:
    import faiss
    from haystack.document_stores import FAISSDocumentStore

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class HaystackDocumentProcessor:
    """Handles document processing using Haystack components."""

    def __init__(
        self,
        document_store: Optional[Any] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        use_faiss: bool = True,
    ):
        """Initialize the processor with Haystack components."""
        logger.info("Initializing Haystack document processor")

        # Initialize converters
        self.text_converter = TextFileToDocument()

        # Initialize preprocessors
        self.cleaner = DocumentCleaner()  # Added for text cleaning
        self.splitter = DocumentSplitter(
            split_by="sentence", split_length=chunk_size, split_overlap=chunk_overlap
        )

        # Store embedding model name
        self.embedding_model = embedding_model

        # Initialize embedders
        self.document_embedder = SentenceTransformersDocumentEmbedder(
            model=embedding_model
        )
        self.text_embedder = SentenceTransformersTextEmbedder(model=embedding_model)

        # Warm up the embedders
        self.document_embedder.warm_up()
        self.text_embedder.warm_up()

        # Get embedding dimension
        embedding_dim = 384  # Default for all-MiniLM-L6-v2

        # Initialize or use provided document store
        if document_store:
            self.document_store = document_store
        elif use_faiss and FAISS_AVAILABLE:
            self.document_store = FAISSDocumentStore(embedding_dim=embedding_dim)
            logger.info("Using FAISS document store")
        else:
            if use_faiss and not FAISS_AVAILABLE:
                logger.warning(
                    "FAISS requested but not available. Using InMemoryDocumentStore instead."
                )
            self.document_store = InMemoryDocumentStore()
            logger.info("Using in-memory document store")

        # Initialize retriever for search
        try:
            self.retriever = InMemoryEmbeddingRetriever(
                document_store=self.document_store
            )
            logger.info("Using embedding retriever")
        except Exception as e:
            logger.warning(
                f"Could not initialize embedding retriever: {e}. Using BM25 retriever instead."
            )
            self.retriever = InMemoryBM25Retriever(document_store=self.document_store)
            logger.info("Using BM25 retriever")

        logger.info(
            f"Initialized processor with model: {embedding_model}, chunk size: {chunk_size}"
        )

    def process_document(self, file_path: Path) -> List[Document]:
        """Process a document file into Haystack Document objects."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing document: {file_path}")

        # Convert document based on file type
        if file_path.suffix.lower() == ".pdf":
            pdf_converter = PyPDFToDocument()
            result = pdf_converter.run(sources=[str(file_path)])
            documents = result["documents"]
        elif file_path.suffix.lower() == ".txt":
            result = self.text_converter.run(file_paths=[str(file_path)])
            documents = result["documents"]
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Clean documents
        cleaned_result = self.cleaner.run(documents=documents)
        cleaned_documents = cleaned_result["documents"]

        # Split documents into chunks
        split_result = self.splitter.run(documents=cleaned_documents)
        chunks = split_result["documents"]

        logger.info(f"Document split into {len(chunks)} chunks")
        return chunks

    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """Generate embeddings for documents."""
        if not documents:
            return []

        logger.info(f"Generating embeddings for {len(documents)} documents")
        result = self.document_embedder.run(documents=documents)
        return result["documents"]

    def add_to_document_store(self, documents: List[Document]) -> None:
        """Add documents to the document store."""
        if not documents:
            return

        logger.info(f"Adding {len(documents)} documents to document store")
        self.document_store.write_documents(documents)
        logger.info(
            f"Document store now contains {len(self.document_store.get_all_documents())} documents"
        )

    def process_and_index_document(self, file_path: Path) -> None:
        """Process a document and add it to the document store with embeddings."""
        chunks = self.process_document(file_path)
        chunks_with_embeddings = self.embed_documents(chunks)
        self.add_to_document_store(chunks_with_embeddings)
        logger.info(f"Document {file_path.name} processed and indexed successfully")

    def search(
        self, query: str, top_k: int = 5, filters: Optional[Dict] = None
    ) -> List[Document]:
        """Search for relevant documents using semantic search."""
        # First, generate embedding for the query text
        query_embedding_result = self.text_embedder.run(text=query)
        query_embedding = query_embedding_result["embedding"]

        # Then use the embedding with the retriever
        result = self.retriever.run(
            query_embedding=query_embedding, top_k=top_k, filters=filters
        )
        documents = result["documents"]
        logger.info(f"Found {len(documents)} documents for query: '{query[:50]}...'")
        return documents

    def save_document_store(self, file_path: str) -> None:
        """Save the document store to disk."""
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        try:
            if FAISS_AVAILABLE and isinstance(self.document_store, FAISSDocumentStore):
                self.document_store.save(file_path)
                logger.info(f"Saved FAISS document store to {file_path}")
            else:
                logger.warning(
                    f"Saving document store not implemented for {type(self.document_store).__name__}"
                )
        except Exception as e:
            logger.error(f"Failed to save document store: {e}")

    @classmethod
    def load_document_store(cls, file_path: str, **kwargs):
        """Load a document store from disk."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS is not available. Cannot load FAISS document store.")
            return cls(**kwargs)

        try:
            document_store = FAISSDocumentStore.load(file_path)
            logger.info(f"Loaded FAISS document store from {file_path}")
            return cls(document_store=document_store, **kwargs)
        except Exception as e:
            logger.error(f"Failed to load document store: {e}")
            logger.info("Initializing new document store")
            return cls(**kwargs)
