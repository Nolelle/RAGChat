#!/usr/bin/env python3
"""
Script to run the CLI.
"""

from .cli import get_cli

app = get_cli()

if __name__ == "__main__":
    app()
