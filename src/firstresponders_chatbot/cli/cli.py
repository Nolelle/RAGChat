"""
Command-line interface module for the FirstRespondersChatbot.

This module provides a command-line interface for users to ask questions
and get responses about first responder procedures and protocols.
"""

import logging
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize rich console for prettier output
console = Console()

# Initialize Typer for command-line interface
app = typer.Typer()


class ChatbotCLI:
    """Command-line interface for the FirstRespondersChatbot."""

    def __init__(self, model_dir: str = "trained-models"):
        """
        Initialize the chatbot CLI with Phi-4 mini.
        """
        self.model_dir = Path(model_dir)
        try:
            self.model, self.tokenizer, self.device = self._load_model()
        except Exception as e:
            console.print(f"[bold red]Critical Error:[/bold red] {str(e)}")
            sys.exit(1)

    def _load_model(self):
        """
        Load the fine-tuned Phi-3 Medium model and tokenizer.

        Returns:
            tuple: (model, tokenizer, device)

        Raises:
            FileNotFoundError: If model directory doesn't exist
            RuntimeError: If model loading fails
        """
        # Check if model exists
        if not self.model_dir.exists():
            error_msg = f"Model directory '{self.model_dir}' not found. Please run train.py first."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Detect hardware
        device = self._setup_device()

        # Load model and tokenizer
        with console.status("[bold green]Loading model...[/bold green]"):
            try:
                # Configure quantization for efficiency
                quantization_config = self._setup_quantization(device)

                # Load model with adapter if available
                model, tokenizer = self._load_model_with_adapter(
                    device, quantization_config
                )

                console.print("[bold green]Model loaded successfully![/bold green]")
                return model, tokenizer, device

            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise RuntimeError(f"Failed to load model: {str(e)}")

    def _setup_device(self):
        """Detect and set up the appropriate device."""
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
        return device

    def _setup_quantization(self, device):
        """Configure quantization based on device."""
        # Skip quantization for MPS (Apple Silicon)
        if device.type == "mps":
            return None

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    def _load_model_with_adapter(self, device, quantization_config):
        """Load model with adapter if available, or full model, or fallback to base model."""
        console.print("Loading model from", self.model_dir)
        model_path = "microsoft/Phi-4-mini-instruct"

        # Try loading with adapter first
        try:
            from peft import PeftModel

            # Load base model with quantization if not on MPS
            if device.type == "mps":
                # Apple Silicon - no quantization
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
            else:
                # CUDA or CPU with quantization
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Check for adapter
            adapter_path = os.path.join(self.model_dir, "adapter")
            if os.path.exists(adapter_path):
                console.print(f"Loading LoRA adapter from {adapter_path}")
                model = PeftModel.from_pretrained(base_model, adapter_path)
                return model, tokenizer

            # Try loading full model
            console.print("No adapter found, trying to load full model")
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_dir),
                quantization_config=(
                    quantization_config if device.type != "mps" else None
                ),
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            return model, tokenizer

        except Exception as e:
            console.print(f"[bold yellow]Warning:[/bold yellow] {str(e)}")
            console.print("Falling back to base Phi-4 mini model")

            # Load base model as fallback
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=(
                    quantization_config if device.type != "mps" else None
                ),
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            return model, tokenizer

    def generate_response(self, question: str) -> str:
        """
        Generate a response using Phi-3 Medium.

        Args:
            question: The question to answer

        Returns:
            str: The generated response

        Raises:
            RuntimeError: If response generation fails
        """
        try:
            # Format question for Phi-3
            prompt = self._create_prompt(question)

            # Tokenize and move to device
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate output with error handling for CUDA OOM
            try:
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
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    # Clear CUDA cache and retry with smaller parameters
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    console.print(
                        "[yellow]Memory issue detected, trying with reduced parameters...[/yellow]"
                    )
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=100,  # Reduced from 250
                            min_new_tokens=20,  # Reduced from 30
                            temperature=0.7,
                            top_p=0.9,
                            top_k=40,
                            repetition_penalty=1.2,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                else:
                    raise  # Re-raise if not OOM error

            # Decode the output
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the assistant's response
            response = self._extract_answer_from_output(full_response, prompt)

            # Clean up GPU memory
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")

    def _extract_answer_from_output(self, full_output: str, prompt: str) -> str:
        """Extract just the assistant's response from the full output."""
        # For Phi-3 format, extract everything after <|assistant|>
        if "<|assistant|>" in full_output:
            response = full_output.split("<|assistant|>")[-1].strip()
            return response

        # Fallback: return everything after the prompt
        response = full_output[len(prompt) :].strip()
        return response

    def _create_prompt(self, question: str) -> str:
        """Create a properly formatted prompt for Phi-3 Medium."""
        # Phi-3 prompt format
        return f"<|system|>\nYou are a knowledgeable first responder assistant designed to provide helpful information about emergency procedures and protocols.\n\nGuidelines:\n1. Answer questions based on your training, being accurate and precise.\n2. Organize your responses with clear structure.\n3. Be honest about your limitations - if you're uncertain, clearly state that.\n4. Avoid making up specific statistics or data you don't have access to.\n5. Focus on providing practical, actionable information when possible.\n6. Explain concepts thoroughly but concisely.\n7. Use professional terminology appropriate for first responder contexts.\n8. Prioritize safety information in your responses.\n<|user|>\n{question}\n<|assistant|>"

    def chat(self):
        """Run an interactive chat session."""
        console.print(
            Panel(
                "[bold blue]First Responders Chatbot[/bold blue]\n\n"
                "Type your questions about emergency procedures and protocols. Type 'exit' to quit.",
                expand=False,
            )
        )

        while True:
            # Get user question
            question = console.input("\n[bold green]You:[/bold green] ")
            if question.lower() in ["exit", "quit", "bye", "goodbye"]:
                console.print("\n[bold blue]Goodbye![/bold blue]")
                break

            # Show spinner while generating
            with Progress() as progress:
                task = progress.add_task("[green]Generating response...", total=None)
                try:
                    answer = self.generate_response(question)
                except Exception as e:
                    console.print(f"[bold red]Error:[/bold red] {str(e)}")
                    continue

            # Print the answer with markdown formatting
            console.print("\n[bold blue]Assistant:[/bold blue]")
            try:
                console.print(Markdown(answer))
            except Exception:
                # Fallback to plain text if markdown parsing fails
                console.print(answer)

    def query(self, question: str):
        """Ask a single question and get the response."""
        console.print(f"\n[bold green]Question:[/bold green] {question}")

        # Show spinner while generating
        with console.status("[green]Generating response..."):
            try:
                answer = self.generate_response(question)
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")
                return

        # Print the answer with markdown formatting
        console.print("\n[bold blue]Answer:[/bold blue]")
        try:
            console.print(Markdown(answer))
        except Exception:
            # Fallback to plain text if markdown parsing fails
            console.print(answer)


def get_cli():
    """Get the Typer CLI app."""
    return app


@app.command()
def chat():
    """Start an interactive chat session with the first responder chatbot."""
    cli = ChatbotCLI()
    cli.chat()


@app.command()
def query(question: str = typer.Argument(..., help="The question to ask the model")):
    """Ask a single question to the chatbot."""
    cli = ChatbotCLI()
    cli.query(question)


if __name__ == "__main__":
    app()
