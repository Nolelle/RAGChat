#!/usr/bin/env python3
"""
Script to set up a Hugging Face token for accessing Llama 2 models.
"""

import os
import sys
import getpass
from pathlib import Path
import logging
import json
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine the token path based on the platform
if sys.platform == "win32":
    token_path = Path.home() / ".huggingface" / "token"
else:
    token_path = Path.home() / ".huggingface" / "token"

# URL for validating token
TOKEN_VALIDATION_URL = "https://huggingface.co/api/whoami"


def validate_token(token):
    """Validate a Hugging Face token."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(TOKEN_VALIDATION_URL, headers=headers)
    return response.status_code == 200


def setup_token():
    """Set up a Hugging Face token for model access."""
    token_dir = token_path.parent
    token_dir.mkdir(parents=True, exist_ok=True)

    # Check if the token already exists
    if token_path.exists():
        with open(token_path, "r") as f:
            token = f.read().strip()

        # Validate existing token
        if validate_token(token):
            logger.info("✅ Existing Hugging Face token is valid.")
            return
        else:
            logger.warning("⚠️ Existing Hugging Face token is invalid.")

    # Prompt for a new token
    print(
        "\nThe Llama 2 model requires authentication to access. Follow these steps:\n"
    )
    print("1. Go to https://huggingface.co/settings/tokens")
    print(
        "2. Request access to the model at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"
    )
    print("3. Create a new token (or use an existing one with read permissions)")
    print("4. Copy the token value\n")

    token = getpass.getpass("Enter your Hugging Face token: ")

    # Validate the token
    if validate_token(token):
        # Save the token
        with open(token_path, "w") as f:
            f.write(token)

        # Set correct permissions on Unix-like systems
        if sys.platform != "win32":
            os.chmod(token_path, 0o600)

        logger.info("✅ Token validated and saved successfully.")
    else:
        logger.error("❌ Invalid token. Please check your token and try again.")
        sys.exit(1)


if __name__ == "__main__":
    setup_token()
