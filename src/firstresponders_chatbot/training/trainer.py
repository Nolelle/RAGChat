#!/usr/bin/env python3
"""
Trainer module for the FirstRespondersChatbot model.
"""

import os
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset, load_dataset
import evaluate
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import random

try:
    import bitsandbytes as bnb  # For 8-bit quantization

    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("bitsandbytes not available, 8-bit quantization will be disabled")

# Correct import for Haystack 2.0
from haystack.components.generators import HuggingFaceAPIGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class to handle the training of the FirstRespondersChatbot model."""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        dataset_path: str = "data/pseudo_data.json",
        output_dir: str = "trained-models/llama3-first-responder",
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        num_train_epochs: int = 2,
        max_seq_length: int = 2048,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 32,
        fp16: bool = True,
        load_in_4bit: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        max_train_samples: Optional[int] = None,  # Parameter for limiting training data
        use_8bit_optimizer: bool = True,  # Control 8-bit optimizer usage
        model_format: str = "llama3",  # Model format to use
    ):
        """
        Initialize the ModelTrainer with parameters suitable for Llama 2.

        Args:
            model_name: Name of the pre-trained model to use
            dataset_path: Path to the training dataset
            output_dir: Directory to save the trained model
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            num_train_epochs: Number of epochs to train for
            max_seq_length: Maximum length of sequences
            weight_decay: Weight decay for regularization
            warmup_ratio: Ratio of total training steps used for learning rate warmup
            gradient_accumulation_steps: Number of steps to accumulate gradients before performing an update
            fp16: Whether to use mixed precision training
            load_in_4bit: Whether to load model in 4-bit precision
            lora_r: Rank of the LoRA update matrices
            lora_alpha: Scaling factor for LoRA
            lora_dropout: Dropout probability for LoRA layers
            max_train_samples: Maximum number of samples to use for training (for faster development)
            use_8bit_optimizer: Whether to use 8-bit optimizers (set to False for Apple Silicon)
            model_format: Format of the model being used (llama2)
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.max_seq_length = max_seq_length
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.load_in_4bit = load_in_4bit
        self.max_train_samples = max_train_samples
        self.use_8bit_optimizer = use_8bit_optimizer and BITSANDBYTES_AVAILABLE
        self.model_format = model_format

        # LoRA parameters
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # Check for hardware acceleration
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info(f"Using Apple Silicon acceleration (MPS)")
            # Force disable 8-bit optimizer on Apple Silicon as it's not compatible
            self.use_8bit_optimizer = False
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("No GPU detected, using CPU (this might be slower)")
            # Force disable 8-bit optimizer on CPU as it's not needed
            self.use_8bit_optimizer = False

    def load_dataset(self) -> Dataset:
        """
        Load and prepare the dataset for training.

        Returns:
            The prepared dataset
        """
        logger.info(f"Loading dataset from {self.dataset_path}")

        try:
            # First, try to read the JSON file directly to handle nested structures
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check if the data has a nested structure with 'train' key
            if isinstance(data, dict) and "train" in data:
                logger.info("Found nested structure with 'train' key")
                train_data = data["train"]

                # Create a temporary file with just the train data
                temp_file = f"{self.dataset_path}.temp"
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(train_data, f)

                # Load the dataset from the temporary file
                dataset_dict = load_dataset("json", data_files=temp_file)

                # Clean up the temporary file
                os.remove(temp_file)

                # Get the dataset from the DatasetDict
                if isinstance(dataset_dict, dict) and "train" in dataset_dict:
                    dataset = dataset_dict["train"]
                else:
                    # If there's no 'train' split, use the first available split
                    first_key = next(iter(dataset_dict))
                    dataset = dataset_dict[first_key]
            else:
                # Try to load the dataset directly using datasets library
                dataset_dict = load_dataset("json", data_files=self.dataset_path)

                # Get the dataset from the DatasetDict
                if isinstance(dataset_dict, dict) and "train" in dataset_dict:
                    dataset = dataset_dict["train"]
                else:
                    # If there's no 'train' split, use the first available split
                    first_key = next(iter(dataset_dict))
                    dataset = dataset_dict[first_key]

            logger.info(f"Loaded dataset with {len(dataset)} examples")

            # Print dataset structure for debugging
            logger.info(f"Dataset column names: {dataset.column_names}")

            # Print a sample of the dataset
            if len(dataset) > 0:
                logger.info(f"Dataset sample: {dataset[0]}")

            # Limit training data if specified (for faster training)
            if (
                self.max_train_samples is not None
                and len(dataset) > self.max_train_samples
            ):
                logger.info(
                    f"Limiting dataset to {self.max_train_samples} examples for faster training"
                )
                # Shuffle dataset before taking a subset to ensure good representation
                dataset = dataset.shuffle(seed=42).select(range(self.max_train_samples))
                logger.info(f"Dataset reduced to {len(dataset)} examples")

            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def format_dataset(self, dataset: Dataset) -> Dataset:
        """
        Format the dataset for Llama 3 causal language model training.

        Args:
            dataset: Input dataset

        Returns:
            Formatted dataset
        """
        logger.info(f"Formatting dataset for {self.model_format}")

        def format_prompt(question, answer):
            """Format the prompt and response in Llama 3's expected format."""
            system_message = "You are a knowledgeable first responder assistant designed to provide helpful information about emergency procedures and protocols."

            # Format with Llama 3 chat template
            if self.model_format == "llama3":
                formatted_text = f"<|system|>\n{system_message}\n<|user|>\n{question}\n<|assistant|>\n{answer}"
            else:
                # Fallback to Llama 2 format
                formatted_text = f"<|im_end|> {system_message}\n<|im_start|>user\n{question}\n<|im_sep|> {answer} <|im_end|>"

            return formatted_text

        def format_samples(examples):
            # Determine input/output columns
            input_col, target_col = self._identify_dataset_columns(dataset)

            # Apply formatting to create prompts in Llama 3 format
            if input_col and target_col:
                texts = [
                    format_prompt(q, a)
                    for q, a in zip(examples[input_col], examples[target_col])
                ]
            elif "text" in examples:
                # Already formatted text
                texts = examples["text"]
            else:
                # Default to "input" and "output" if they exist as fallback
                texts = [
                    format_prompt(q, a)
                    for q, a in zip(
                        examples.get("input", []),
                        examples.get("output", examples.get("input", [])),
                    )
                ]

            return {"text": texts}

        # Apply formatting to dataset
        formatted_dataset = dataset.map(
            format_samples, batched=True, remove_columns=dataset.column_names
        )

        logger.info(f"Formatted dataset with {len(formatted_dataset)} examples")

        # Show a sample of the formatted dataset
        if len(formatted_dataset) > 0:
            logger.info(f"Formatted sample: {formatted_dataset[0]}")

        return formatted_dataset

    def _load_and_prepare_model(self):
        """
        Load and prepare the Llama 2 model with quantization and LoRA.

        Returns:
            The prepared model and tokenizer
        """
        logger.info(f"Loading model: {self.model_name}")

        # Skip 4-bit loading when using MPS (Apple Silicon)
        use_4bit = self.load_in_4bit and self.device.type != "mps"

        quantization_config = None
        if use_4bit:
            logger.info("Using 4-bit quantization for model loading")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        try:
            # Load tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # Make sure padding token is set
            if tokenizer.pad_token is None:
                if tokenizer.eos_token:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "<pad>"})

            # Load model with appropriate settings
            if self.device.type == "mps":
                logger.info("Loading model for MPS (Apple Silicon) with optimizations")
                # Clear MPS memory before loading model
                torch.mps.empty_cache()

                # For Apple Silicon MPS backend, use optimized loading
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
                )

                # Move model to MPS device after loading
                model = model.to(self.device)

                # Apply memory-efficient settings for Apple Silicon
                model.config.use_cache = False  # Disable KV cache to save memory

                # Enable gradient checkpointing to reduce memory usage
                if hasattr(model, "gradient_checkpointing_enable"):
                    logger.info("Enabling gradient checkpointing for MPS")
                    model.gradient_checkpointing_enable()
            else:
                # Standard loading for CUDA/CPU with optional quantization
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",  # Let HF decide device mapping
                    trust_remote_code=True,
                )

            # Prepare the model for LoRA training
            logger.info("Preparing model for LoRA training")

            # If using 4-bit quantization, prepare model for kbit training
            if use_4bit:
                model = prepare_model_for_kbit_training(model)

            # Setup LoRA configuration
            target_modules = self._get_lora_target_modules(model)

            logger.info(f"Using LoRA with target modules: {target_modules}")

            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

            # Apply LoRA to the model
            model = get_peft_model(model, lora_config)

            # Print number of trainable parameters
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(
                f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%} of {total_params:,} total parameters)"
            )

            return model, tokenizer

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _get_lora_target_modules(self, model):
        """Determine the appropriate target modules for LoRA based on model architecture."""
        # Check if it's a Llama model
        if "llama" in self.model_name.lower():
            if self.model_format == "llama3":
                logger.info("Detected Llama 3 model, using optimized target modules")
                # Target the attention modules for Llama 3
                return [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            else:
                logger.info("Detected Llama 2 model, using optimized target modules")
                # Target the attention modules for Llama 2
                return ["q_proj", "k_proj", "v_proj", "o_proj"]

        # Generic approach for other models - find linear layers by partial name match
        target_modules = []
        module_types = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "attention",
            "mlp",
        ]

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                for module_type in module_types:
                    if module_type in name.lower():
                        target_modules.append(name.split(".")[-1])
                        break

        # Remove duplicates and return
        target_modules = list(set(target_modules))

        if not target_modules:
            # If no modules found, fall back to default targets for transformer models
            logger.warning("No target modules found, using default targets for Llama 2")
            target_modules = ["q_proj", "v_proj"]

        return target_modules

    def train(
        self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None
    ) -> None:
        """
        Train the model with the given dataset.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
        """
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Training {self.model_format.upper()} model")
        logger.info(f"Training parameters:")
        logger.info(f"  - Batch size: {self.batch_size}")
        logger.info(f"  - Learning rate: {self.learning_rate}")
        logger.info(f"  - Epochs: {self.num_train_epochs}")
        logger.info(f"  - Max sequence length: {self.max_seq_length}")
        logger.info(
            f"  - Gradient accumulation steps: {self.gradient_accumulation_steps}"
        )
        logger.info(f"  - FP16: {self.fp16}")

        # Apply Apple Silicon specific optimizations
        if self.device.type == "mps":
            logger.info("Applying Apple Silicon (MPS) specific optimizations")
            # Clear MPS memory
            torch.mps.empty_cache()

        # Load model and tokenizer
        model, tokenizer = self._load_and_prepare_model()

        # Handle Apple Silicon device mapping explicitly
        if self.device.type == "mps":
            model = model.to(self.device)

        # Format the dataset to use the appropriate template
        logger.info(f"Formatting dataset for {self.model_format.upper()}")

        # Identify input and target columns
        input_col, target_col = self._identify_dataset_columns(train_dataset)

        # Format the datasets
        formatted_train_dataset = self._format_dataset_for_llama(
            train_dataset, input_col, target_col
        )

        # Handle evaluation dataset if provided
        if eval_dataset is not None:
            formatted_eval_dataset = self._format_dataset_for_llama(
                eval_dataset, input_col, target_col
            )
        else:
            formatted_eval_dataset = None

        # Set up a tokenization function to apply to the formatted data
        def tokenize_function(examples):
            # Ensure we're working with flat strings, not nested lists
            texts = examples["text"]
            if (
                isinstance(texts, list)
                and len(texts) > 0
                and isinstance(texts[0], list)
            ):
                texts = [t[0] if len(t) > 0 else "" for t in texts]

            # Tokenize the texts
            tokenized = tokenizer(
                texts,
                truncation=True,
                max_length=self.max_seq_length,
                padding=(
                    "max_length" if self.device.type == "mps" else False
                ),  # Always pad for MPS
                return_tensors=None,  # Return as lists, not tensors
            )

            # Set labels to input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        # Apply tokenization to the training dataset
        logger.info("Tokenizing training dataset")
        tokenized_train_dataset = formatted_train_dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        # Apply tokenization to the evaluation dataset if available
        tokenized_eval_dataset = None
        if formatted_eval_dataset is not None:
            logger.info("Tokenizing evaluation dataset")
            tokenized_eval_dataset = formatted_eval_dataset.map(
                tokenize_function, batched=True, remove_columns=["text"]
            )

        # Apple Silicon specific data prefetching
        if self.device.type == "mps":
            logger.info("Setting up Apple Silicon optimized data formats")
            # Use smaller prefetch factor for MPS
            tokenized_train_dataset = tokenized_train_dataset.with_format(
                "torch", device=self.device
            )
            if tokenized_eval_dataset:
                tokenized_eval_dataset = tokenized_eval_dataset.with_format(
                    "torch", device=self.device
                )

        # Data collator for causal language modeling
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            num_train_epochs=self.num_train_epochs,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_strategy="steps",
            logging_steps=10,
            evaluation_strategy="epoch" if formatted_eval_dataset else "no",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True if formatted_eval_dataset else False,
            report_to="tensorboard",
            seed=42,
            dataloader_pin_memory=(
                False if self.device.type == "mps" else True
            ),  # Disable pin memory for MPS
        )

        # MPS-specific mixed precision settings
        if self.device.type == "mps":
            logger.info("Using MPS-specific settings for mixed precision")
            training_args.fp16 = True
            training_args.optim = "adamw_torch"  # Use standard optimizer for MPS
            # Disable 8-bit optimizations on MPS
            self.use_8bit_optimizer = False
        else:
            training_args.fp16 = self.fp16
            training_args.optim = (
                "adamw_torch"
                if self.device.type == "mps"
                else "adamw_8bit" if self.use_8bit_optimizer else "adamw_torch"
            )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # Train the model
        logger.info("Starting training...")
        trainer.train()

        # Save the model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        logger.info("Training completed successfully!")

    def _format_dataset_for_llama(self, dataset, input_col, target_col):
        """Format dataset specifically for Llama model structure."""

        def format_example(example):
            system_prompt = "You are a knowledgeable first responder assistant designed to provide helpful information about emergency procedures and protocols."
            user_input = example[input_col]
            assistant_output = example[target_col]

            if self.model_format == "llama3":
                return {
                    "text": f"<|system|>\n{system_prompt}\n<|user|>\n{user_input}\n<|assistant|>\n{assistant_output}"
                }
            else:
                # Llama 2 format
                return {
                    "text": f"<|im_end|> {system_prompt}\n<|im_start|>user\n{user_input}\n<|im_sep|> {assistant_output} <|im_end|>"
                }

        return dataset.map(format_example, remove_columns=dataset.column_names)

    def _identify_dataset_columns(self, dataset: Dataset) -> Tuple[str, str]:
        """
        Identify input and target columns from the dataset.

        Args:
            dataset: The dataset to analyze

        Returns:
            Tuple of (input_column_name, target_column_name)
        """
        # List of common names for input and target columns
        input_names = [
            "input",
            "inputs",
            "source",
            "question",
            "context",
            "premise",
            "instruction",
            "query",
            "prompt",
        ]
        target_names = [
            "target",
            "targets",
            "output",
            "outputs",
            "answer",
            "response",
            "label",
            "labels",
            "hypothesis",
            "completion",
        ]

        # Get column names from dataset
        columns = dataset.column_names

        # Try to find input column
        input_col = None
        for name in input_names:
            if name in columns:
                input_col = name
                break

        # Try to find target column
        target_col = None
        for name in target_names:
            if name in columns:
                target_col = name
                break

        # If we still haven't found the columns, make a best guess
        if input_col is None or target_col is None:
            if len(columns) == 2:
                # If there are exactly two columns, assume the first is input
                # and the second is target
                input_col = columns[0]
                target_col = columns[1]
            else:
                # Try to find columns with common patterns
                for col in columns:
                    col_lower = col.lower()
                    if any(name in col_lower for name in input_names):
                        input_col = col
                    elif any(name in col_lower for name in target_names):
                        target_col = col

        # Raise an error if we couldn't identify the columns
        if input_col is None or target_col is None:
            raise ValueError(
                f"Could not identify input and target columns in dataset. "
                f"Available columns: {columns}"
            )

        return input_col, target_col

    def run(self) -> None:
        """
        Run the complete training pipeline.
        """
        # Load dataset
        dataset = self.load_dataset()

        # Split dataset into train and evaluation sets
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]

        # Print key information about the training setup
        print(
            f"Training on device: {'CUDA' if torch.cuda.is_available() else 'MPS' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'CPU'}"
        )
        print(f"Model: {self.model_name}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Evaluation samples: {len(eval_dataset)}")
        print(f"Max sequence length: {self.max_seq_length}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Starting training now...")

        # Train the model
        self.train(train_dataset=train_dataset, eval_dataset=eval_dataset)


if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        output_dir="trained-models/llama3-first-responder",
    )
    trainer.run()
