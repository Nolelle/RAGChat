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
        "--document-store-dir",
        type=str,
        default="uploads",
        help="Directory to store uploaded files",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--use-mps",
        action="store_true",
        default=True,
        help="Use MPS (Apple Silicon) when available",
    )
    args = parser.parse_args()

    # Initialize RAG system
    rag_system = RAGSystem(
        model_name_or_path="trained-models",
        document_store_dir=args.document_store_dir,
        top_k=args.top_k,
        use_mps=args.use_mps,
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
