#!/usr/bin/env python3
"""
Server script for the FirstRespondersChatbot web interface.
"""

import sys
import os
import logging
from pathlib import Path

# Check if token exists
token_path = Path.home() / ".huggingface" / "token"
if not token_path.exists():
    print("Hugging Face token not found.")
    print("The Llama 2 model requires authentication to access.")
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

    # Run the server
    logger.info("Starting FirstResponders Chatbot server with Llama 2...")
    run_server(host="0.0.0.0", port=8000, model_name="meta-llama/Llama-2-7b-chat-hf")

except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please make sure all dependencies are installed.")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error starting server: {e}")
    sys.exit(1)
