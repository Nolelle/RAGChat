#!/usr/bin/env python3
"""
Main entry point for the FirstRespondersChatbot package.

This module provides a command-line interface to run the different components
of the FirstRespondersChatbot project.
"""

import argparse
import sys
import importlib


def main():
    """Main entry point for the FirstRespondersChatbot package."""
    parser = argparse.ArgumentParser(description="FirstRespondersChatbot")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess documents")
    preprocess_parser.add_argument(
        "--docs-dir",
        type=str,
        default="docs",
        help="Directory containing the documents to preprocess",
    )
    preprocess_parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save the preprocessed documents",
    )

    # Create dataset command
    dataset_parser = subparsers.add_parser(
        "create-dataset", help="Create a dataset for fine-tuning"
    )
    dataset_parser.add_argument(
        "--input-file",
        type=str,
        default="data/preprocessed_data.json",
        help="Path to the preprocessed data file",
    )
    dataset_parser.add_argument(
        "--output-file",
        type=str,
        default="data/pseudo_data.json",
        help="Path to save the generated dataset",
    )
    dataset_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training",
    )
    dataset_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio of data to use for validation",
    )
    dataset_parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Ratio of data to use for testing"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Fine-tune the model")
    # Add train arguments here

    # CLI command
    cli_parser = subparsers.add_parser("cli", help="Run the CLI")
    cli_subparsers = cli_parser.add_subparsers(
        dest="cli_command", help="CLI command to run"
    )
    chat_parser = cli_subparsers.add_parser(
        "chat", help="Start an interactive chat session"
    )
    query_parser = cli_subparsers.add_parser("query", help="Ask a single question")
    query_parser.add_argument(
        "question", type=str, help="The question to ask the model"
    )

    # Server command
    server_parser = subparsers.add_parser("server", help="Run the RAG server")
    server_parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to"
    )
    server_parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to"
    )
    server_parser.add_argument(
        "--debug", action="store_true", help="Run the server in debug mode"
    )

    args = parser.parse_args()

    if args.command == "preprocess":
        from firstresponders_chatbot.preprocessing.preprocessor import (
            DocumentPreprocessor,
        )

        preprocessor = DocumentPreprocessor(
            docs_dir=args.docs_dir, output_dir=args.output_dir
        )
        preprocessor.run()
    elif args.command == "create-dataset":
        from firstresponders_chatbot.training.dataset_creator import DatasetCreator

        creator = DatasetCreator(
            input_file=args.input_file,
            output_file=args.output_file,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        creator.run()
    elif args.command == "train":
        # Import and run train module
        pass
    elif args.command == "cli":
        from firstresponders_chatbot.cli.cli import ChatbotCLI

        cli = ChatbotCLI()
        if args.cli_command == "chat":
            cli.chat()
        elif args.cli_command == "query":
            cli.query(args.question)
        else:
            cli_parser.print_help()
    elif args.command == "server":
        from firstresponders_chatbot.rag.rag_system import RAGSystem
        from firstresponders_chatbot.rag.server import RAGServer

        rag_system = RAGSystem()
        server = RAGServer(
            rag_system=rag_system,
            host=args.host,
            port=args.port,
            debug=args.debug,
        )
        server.run()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
