#!/usr/bin/env python3
"""
Script to run the RAG server.
"""

import argparse
import os
from .rag_system import RAGSystem
from .server import RAGServer


def main():
    """Main function to run the RAG server."""
    parser = argparse.ArgumentParser(
        description="Run the RAG server for the FirstRespondersChatbot"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 8000)),
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run the server in debug mode"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="tinyllama-1.1b-first-responder-fast",
        help="Directory containing the fine-tuned model",
    )
    parser.add_argument(
        "--uploads-dir",
        type=str,
        default="uploads",
        help="Directory to store uploaded files",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Name of the embedding model to use",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of documents to retrieve"
    )
    args = parser.parse_args()

    # Initialize RAG system
    rag_system = RAGSystem(
        model_dir=args.model_dir,
        uploads_dir=args.uploads_dir,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
    )

    # Initialize and run server
    server = RAGServer(
        rag_system=rag_system,
        host=args.host,
        port=args.port,
        debug=args.debug,
    )
    server.run()


if __name__ == "__main__":
    main()
