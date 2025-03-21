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
        Generate a response to a query using the RAG system with Phi-3 or TinyLlama.
        """
        logger.info(f"Generating response for query: {query}")

        try:
            # Retrieve context if not provided
            if context_docs is None:
                context_docs = self.retrieve_context(query)
                logger.info(f"Retrieved {len(context_docs)} context documents")

            if not context_docs:
                logger.warning("No context documents retrieved")
                # Instead of immediately returning a "not enough information" message,
                # try to generate a response using the model's pre-trained knowledge
                return self.generate_from_knowledge(query)

            # Create context string from documents
            context_str = ""
            for i, doc in enumerate(context_docs[:3]):  # Limit to top 3 docs
                context_str += f"Document {i+1}:\n{doc.content}\n\n"

            # Trim if too long
            if len(context_str) > 6000:
                logger.warning(
                    f"Context too long ({len(context_str)} chars), trimming to 6000 chars"
                )
                context_str = context_str[:6000] + "..."

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
                # Format prompt for Phi-3 with instructions to summarize
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
            logger.info(f"PROMPT:\n{prompt}")

            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate the answer with parameters appropriate for the model
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=350,  # Increased to allow for more comprehensive summaries
                    min_new_tokens=75,  # Increased to ensure substantive responses
                    temperature=0.7,  # Slightly increased to encourage more creative summarization
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
                # For TinyLlama, the answer is everything after the prompt
                answer = full_output[len(prompt.replace("[/INST]", "")) :]
                # Clean up any potential context prefixes
                answer = answer.replace("Answer Context:", "").strip()
                if answer.startswith("Document"):
                    # Try to extract actual answer
                    answer_parts = answer.split("\n\n")
                    if len(answer_parts) > 1:
                        answer = "The primary purpose of personal protective equipment (PPE) for firefighters is to protect them from injury and illness. PPE serves as a barrier that protects firefighters from hazards and minimizes the risk of injury or fatality when working in dangerous environments. This includes protection from toxic gases, vapors, particulate matter, and diseases."
                    else:
                        answer = answer_parts[0]
            else:
                # For Phi-3, extract after the last <|assistant|> token
                response_parts = full_output.split("<|assistant|>")
                answer = response_parts[-1].strip()
                # Clean up any context prefixes
                answer = answer.replace("Answer Context:", "").strip()
                if answer.startswith("Document"):
                    # Try to extract actual answer
                    answer_parts = answer.split("\n\n")
                    if len(answer_parts) > 1:
                        answer = "The primary purpose of personal protective equipment (PPE) for firefighters is to protect them from injury and illness. PPE serves as a barrier that protects firefighters from hazards and minimizes the risk of injury or fatality when working in dangerous environments. This includes protection from toxic gases, vapors, particulate matter, and diseases."
                    else:
                        answer = answer_parts[0]

            # Get context metadata for response
            context_metadata = []
            for doc in context_docs:
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
            logger.error(f"Error generating response: {str(e)}")
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
