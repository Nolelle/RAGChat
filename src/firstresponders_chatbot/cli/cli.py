"""
Command-line interface module for the FirstRespondersChatbot.

This module provides a command-line interface for users to ask questions
and get responses about first responder procedures and protocols.
"""

import logging
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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

    def __init__(self, model_dir: str = "phi-3-mini-first-responder"):
        """
        Initialize the chatbot CLI with Phi-3.
        """
        self.model_dir = Path(model_dir)
        self.model, self.tokenizer, self.device = self._load_model()

    def _load_model(self):
        """
        Load the fine-tuned Phi-3 model and tokenizer.
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

        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # Load Phi-3 model
        console.print("Loading model from", self.model_dir)

        # Try loading with adapter first
        try:
            from peft import PeftModel

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct",
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Check for adapter
            adapter_path = os.path.join(self.model_dir, "adapter")
            if os.path.exists(adapter_path):
                console.print(f"Loading LoRA adapter from {adapter_path}")
                model = PeftModel.from_pretrained(base_model, adapter_path)
            else:
                console.print("No adapter found, trying to load full model")
                model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_dir),
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
        except Exception as e:
            console.print(f"[bold yellow]Warning:[/bold yellow] {str(e)}")
            console.print("Falling back to base Phi-3 model")
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct",
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer, device

    def generate_response(self, question: str) -> str:
        """
        Generate a response using Phi-3.
        """
        # Format question for Phi-3
        prompt = f"""<|system|>
You are a first responders chatbot designed to provide accurate information about emergency procedures, protocols, and best practices. Focus on delivering complete, accurate responses that address the core purpose and function of equipment or procedures. When discussing protective equipment, prioritize explaining its primary protective purpose before maintenance details.
<|user|>
{question}
<|assistant|>"""

        # Tokenize and move to device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=250,
                min_new_tokens=30,
                temperature=0.6,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode the output
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant's response
        response_parts = full_response.split("<|assistant|>")
        response = response_parts[-1].strip()

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
