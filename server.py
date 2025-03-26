#!/usr/bin/env python3
"""
Server script for the FirstRespondersChatbot web interface.
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# Set up argument parser
parser = argparse.ArgumentParser(description="Run the FirstRespondersChatbot server")
parser.add_argument(
    "--model",
    type=str,
    default="tinyllama-1.1b-first-responder-fast",
    choices=[
        "tinyllama-1.1b-first-responder-fast",
        "llama-3.1-1b-first-responder",
        "meta-llama/Llama-2-7b-chat-hf",
    ],
    help="Model to use for the chatbot (default: tinyllama-1.1b-first-responder-fast)",
)
parser.add_argument(
    "--host",
    type=str,
    default="0.0.0.0",
    help="Host to run the server on (default: 0.0.0.0)",
)
parser.add_argument(
    "--port", type=int, default=8000, help="Port to run the server on (default: 8000)"
)
args = parser.parse_args()

# Check if token exists
token_path = Path.home() / ".huggingface" / "token"
if not token_path.exists() and "meta-llama" in args.model:
    print("Hugging Face token not found.")
    print("The Llama models require authentication to access.")
    print("Please run: python setup_token.py")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    # Import the server module
    from src.firstresponders_chatbot.rag.run_server import run_server

    model_name = args.model
    model_display_name = {
        "tinyllama-1.1b-first-responder-fast": "TinyLlama 1.1B",
        "llama-3.1-1b-first-responder": "Llama 3.1 1B",
        "meta-llama/Llama-2-7b-chat-hf": "Llama 2 7B",
    }.get(model_name, model_name)

    # Run the server
    logger.info(f"Starting FirstResponders Chatbot server with {model_display_name}...")
    run_server(host=args.host, port=args.port, model_name=model_name)

except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please make sure all dependencies are installed.")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error starting server: {e}")
    sys.exit(1)
