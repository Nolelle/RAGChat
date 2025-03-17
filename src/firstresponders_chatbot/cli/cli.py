"""
Command-line interface module for the FirstRespondersChatbot.

This module provides a command-line interface for users to ask questions
and get responses about first responder procedures and protocols.
"""

import logging
import os
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rich console for prettier output
console = Console()

# Initialize Typer for command-line interface
app = typer.Typer()


class ChatbotCLI:
    """Command-line interface for the FirstRespondersChatbot."""

    def __init__(self, model_dir: str = "flan-t5-first-responder"):
        """
        Initialize the chatbot CLI.

        Args:
            model_dir: Directory containing the fine-tuned model
        """
        self.model_dir = Path(model_dir)
        self.model, self.tokenizer, self.device = self._load_model()

    def _load_model(self):
        """
        Load the fine-tuned model and tokenizer.

        This function checks for available hardware acceleration (Apple Silicon MPS
        or NVIDIA CUDA) and loads the model onto the appropriate device.

        Returns:
            tuple: (model, tokenizer, device)
        """
        # Check if model exists
        if not self.model_dir.exists():
            console.print(
                "[bold red]Error:[/bold red] Model directory not found. Please run train.py first."
            )
            raise typer.Exit(code=1)

        # Detect hardware
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            console.print(f"[bold green]Using Apple Silicon acceleration[/bold green]")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            console.print(
                f"[bold green]Using NVIDIA GPU: {torch.cuda.get_device_name(0)}[/bold green]"
            )
        else:
            device = torch.device("cpu")
            console.print(
                "[yellow]No GPU detected, using CPU (this might be slower)[/yellow]"
            )

        # Load model and tokenizer
        console.print("Loading model from", self.model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

        # Move model to device
        model = model.to(device)

        return model, tokenizer, device

    def generate_response(self, question: str) -> str:
        """
        Generate a response to the user's question.

        Args:
            question: User's question

        Returns:
            str: Generated response
        """
        # Format question as in training
        input_text = (
            f"question: {question}"
            if not question.startswith("question: ")
            else question
        )

        # Tokenize and move to device
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(
            self.device
        )

        # Generate output
        outputs = self.model.generate(
            input_ids,
            max_length=512,  # Maximum length of the generated response
            num_beams=4,  # Beam search for better quality responses
            temperature=0.7,  # Add some randomness (values closer to 0 are more deterministic)
            no_repeat_ngram_size=2,  # Avoid repeating the same phrases
        )

        # Decode and return response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def chat(self):
        """
        Start an interactive chat session with the first responders chatbot.

        This mode allows you to ask multiple questions in sequence, similar to
        having a conversation with the chatbot.
        """
        console.print(
            Panel.fit(
                "[bold blue]First Responders Chatbot[/bold blue]\n"
                "Type your questions about first aid, emergency procedures, or disaster response.\n"
                "Type 'exit' or 'quit' to end the session."
            )
        )

        # Interactive loop
        while True:
            # Get user input
            question = console.input("\n[bold green]You:[/bold green] ")

            # Check if user wants to exit
            if question.lower() in ["exit", "quit", "q"]:
                console.print("\n[bold blue]Goodbye![/bold blue]")
                break

            try:
                # Show thinking indicator
                with console.status(
                    "[bold yellow]Generating response...[/bold yellow]"
                ):
                    response = self.generate_response(question)

                # Display response as markdown for better formatting
                console.print("\n[bold blue]Bot:[/bold blue]")
                console.print(Markdown(response))

            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")

    def query(self, question: str):
        """
        Ask a single question and get a response (non-interactive mode).

        This mode is useful for scripting or when you just need a quick answer
        without starting an interactive session.

        Args:
            question: The question to ask the chatbot
        """
        try:
            # Show thinking indicator
            with console.status("[bold yellow]Generating response...[/bold yellow]"):
                response = self.generate_response(question)

            # Display response
            console.print(Markdown(response))

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


def get_cli():
    """Get the Typer CLI app with commands registered."""
    cli = ChatbotCLI()

    @app.command()
    def chat():
        """
        Start an interactive chat session with the first responders chatbot.
        """
        cli.chat()

    @app.command()
    def query(
        question: str = typer.Argument(..., help="The question to ask the model")
    ):
        """
        Ask a single question and get a response (non-interactive mode).
        """
        cli.query(question)

    return app
