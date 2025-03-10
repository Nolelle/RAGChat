import typer
from pathlib import Path
import logging
from rich.console import Console
from rich.logging import RichHandler
from rich import print
from typing import List, Optional
import os

from src.data_processing import HaystackDocumentProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("haystack-cli")

# Create Typer app
app = typer.Typer()
console = Console()

# Global paths
DEFAULT_DATA_DIR = Path("data")
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_PROCESSED_DIR = DEFAULT_DATA_DIR / "processed"
DEFAULT_EMBEDDINGS_DIR = DEFAULT_DATA_DIR / "embeddings"

# Ensure directories exist
DEFAULT_RAW_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Global processor (lazy-loaded)
_processor = None


def get_processor():
    """Get or initialize document processor."""
    global _processor
    if _processor is None:
        _processor = HaystackDocumentProcessor()
    return _processor


@app.command()
def process_document(
    file_path: Path = typer.Argument(..., help="Path to the document to process")
):
    """Process a document and add it to the knowledge base."""
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return

    logger.info(f"Processing document: {file_path}")

    # Process and index document
    processor = get_processor()
    processor.process_and_index_document(file_path)

    logger.info(f"Document {file_path.name} processed and indexed successfully")


@app.command()
def process_directory(
    directory: Path = typer.Argument(
        ..., help="Directory containing documents to process"
    )
):
    """Process all documents in a directory."""
    if not directory.exists() or not directory.is_dir():
        logger.error(f"Directory not found: {directory}")
        return

    files = list(directory.glob("*.txt")) + list(directory.glob("*.pdf"))
    logger.info(f"Found {len(files)} documents to process")

    processor = get_processor()
    for file_path in files:
        logger.info(f"Processing {file_path.name}")
        processor.process_and_index_document(file_path)

    logger.info(f"All {len(files)} documents processed and indexed successfully")


@app.command()
def query(
    query_text: str = typer.Argument(..., help="Query to search for"),
    top_k: int = typer.Option(3, help="Number of results to return"),
):
    """Search for documents relevant to a query."""
    processor = get_processor()
    results = processor.search(query_text, top_k=top_k)

    if not results:
        print("[bold red]No relevant documents found.[/bold red]")
        return

    print(f"[bold green]Found {len(results)} relevant documents:[/bold green]\n")
    for i, doc in enumerate(results):
        print(
            f"[bold blue]{i+1}. From {doc.meta.get('source', 'Unknown')} (Score: {doc.score:.4f})[/bold blue]"
        )
        print(f"{doc.content[:500]}...\n")


@app.command()
def chat():
    """Interactive chat mode."""
    print("[bold green]First Responder RAG Chatbot[/bold green]")
    print("Type your questions or 'exit' to quit.\n")

    processor = get_processor()

    while True:
        query_text = console.input("[bold blue]Question:[/bold blue] ")
        if query_text.lower() in ("exit", "quit", "q"):
            break

        if not query_text.strip():
            continue

        results = processor.search(query_text)

        if not results:
            print(
                "[bold red]No relevant information found in the knowledge base.[/bold red]"
            )
            continue

        # In a full implementation, you would use these results with an LLM to generate a response
        # For now, we'll just show the retrieved chunks
        print("\n[bold green]Relevant information:[/bold green]")
        for i, doc in enumerate(results):
            print(
                f"[bold blue]{i+1}. From {doc.meta.get('source', 'Unknown')} (Score: {doc.score:.4f})[/bold blue]"
            )
            print(f"{doc.content[:500]}...\n")


if __name__ == "__main__":
    app()
