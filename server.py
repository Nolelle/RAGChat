#!/usr/bin/env python3
"""
Script to run the RAG server.
"""

import sys
import os
from src.firstresponders_chatbot.rag.rag_system import RAGSystem
from src.firstresponders_chatbot.rag.server import RAGServer


def main():
    """Main function to run the RAG server."""
    # Initialize RAG system
    rag_system = RAGSystem()

    # Initialize and run server
    port = int(os.environ.get("PORT", 8000))
    server = RAGServer(
        rag_system=rag_system,
        port=port,
        debug=True,
    )
    server.run()


if __name__ == "__main__":
    main()
