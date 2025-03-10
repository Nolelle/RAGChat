from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
from haystack import Document
from haystack.components.converters import PDFToTextConverter, TextFileToTextConverter
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.document_stores.types import FilterType

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
        """Initialize the processor with Haystack components.

        Args:
            document_store: Optional document store to use
            embedding_model: Name of the sentence transformer model for embeddings
            chunk_size: Maximum size of chunks in characters
            chunk_overlap: Overlap between chunks in characters
            use_faiss: Whether to use FAISS document store for persistence
        """
        logger.info("Initializing Haystack document processor")

        # Initialize converters for different file types
        self.pdf_converter = PDFToTextConverter()
        self.text_converter = TextFileToTextConverter()

        # Initialize document splitter
        self.splitter = DocumentSplitter(
            split_by="sentence", split_length=chunk_size, split_overlap=chunk_overlap
        )

        # Initialize document embedder
        self.document_embedder = SentenceTransformersDocumentEmbedder(
            model_name_or_path=embedding_model
        )
        self.text_embedder = SentenceTransformersTextEmbedder(
            model_name_or_path=embedding_model
        )

        # Initialize or use provided document store
        if document_store:
            self.document_store = document_store
        elif use_faiss:
            self.document_store = FAISSDocumentStore(
                embedding_dim=self.document_embedder.embedding_dim
            )
        else:
            self.document_store = InMemoryDocumentStore()

        logger.info(
            f"Initialized processor with model: {embedding_model}, chunk size: {chunk_size}"
        )

    def process_document(self, file_path: Path) -> List[Document]:
        """Process a document file into Haystack Document objects.

        Args:
            file_path: Path to the document file

        Returns:
            List of processed Haystack Document objects
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing document: {file_path}")

        # Convert document to text based on file type
        if file_path.suffix.lower() == ".pdf":
            result = self.pdf_converter.run(file_paths=[str(file_path)])
        elif file_path.suffix.lower() == ".txt":
            result = self.text_converter.run(file_paths=[str(file_path)])
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Prepare document for splitting
        documents = [
            Document(
                content=text,
                meta={"source": file_path.name, "file_path": str(file_path)},
            )
            for text in result["texts"]
        ]

        # Split document into chunks
        result = self.splitter.run(documents=documents)
        chunks = result["documents"]

        logger.info(f"Document split into {len(chunks)} chunks")
        return chunks

    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """Generate embeddings for documents.

        Args:
            documents: List of Document objects

        Returns:
            Documents with embeddings
        """
        if not documents:
            return []

        logger.info(f"Generating embeddings for {len(documents)} documents")
        result = self.document_embedder.run(documents=documents)
        return result["documents"]

    def add_to_document_store(self, documents: List[Document]) -> None:
        """Add documents to the document store.

        Args:
            documents: List of Document objects to add
        """
        if not documents:
            return

        logger.info(f"Adding {len(documents)} documents to document store")
        self.document_store.write_documents(documents)
        logger.info(
            f"Document store now contains {len(self.document_store.get_all_documents())} documents"
        )

    def process_and_index_document(self, file_path: Path) -> None:
        """Process a document and add it to the document store with embeddings.

        Args:
            file_path: Path to the document file
        """
        # Process document into chunks
        chunks = self.process_document(file_path)

        # Generate embeddings
        chunks_with_embeddings = self.embed_documents(chunks)

        # Add to document store
        self.add_to_document_store(chunks_with_embeddings)

        logger.info(f"Document {file_path.name} processed and indexed successfully")

    def search(
        self, query: str, top_k: int = 5, filters: Optional[FilterType] = None
    ) -> List[Document]:
        """Search for relevant documents using semantic search.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
            List of Document objects
        """
        # Generate query embedding
        query_embedding = self.text_embedder.run(text=query)["embedding"]

        # Search in document store
        documents = self.document_store.search_documents_by_embedding(
            query_embedding=query_embedding, filters=filters, top_k=top_k
        )

        logger.info(f"Found {len(documents)} documents for query: '{query[:50]}...'")
        return documents

    def save_document_store(self, file_path: str) -> None:
        """Save the document store to disk.

        Args:
            file_path: Path to save the document store
        """
        if isinstance(self.document_store, FAISSDocumentStore):
            self.document_store.save(file_path)
            logger.info(f"Saved FAISS document store to {file_path}")
        else:
            logger.warning(
                f"Saving document store not implemented for {type(self.document_store).__name__}"
            )

    @classmethod
    def load_document_store(cls, file_path: str, **kwargs):
        """Load a document store from disk.

        Args:
            file_path: Path to the saved document store
            **kwargs: Additional arguments for the processor

        Returns:
            HaystackDocumentProcessor instance with loaded document store
        """
        try:
            document_store = FAISSDocumentStore.load(file_path)
            logger.info(f"Loaded FAISS document store from {file_path}")
            return cls(document_store=document_store, **kwargs)
        except Exception as e:
            logger.error(f"Failed to load document store: {e}")
            logger.info("Initializing new document store")
            return cls(**kwargs)
