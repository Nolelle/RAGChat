#!/usr/bin/env python3
"""
Script to run the CLI.
"""

import sys
from src.firstresponders_chatbot.cli.cli import get_cli

app = get_cli()

if __name__ == "__main__":
    app()
