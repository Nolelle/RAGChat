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
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        dataset_path: str = "data/pseudo_data.json",
        output_dir: str = "phi-3-mini-first-responder",
        batch_size: int = 1,
        learning_rate: float = 2e-4,
        num_train_epochs: int = 3,
        max_seq_length: int = 2048,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 16,
        fp16: bool = True,
        load_in_4bit: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        max_train_samples: Optional[
            int
        ] = None,  # Added parameter for limiting training data
    ):
        """
        Initialize the ModelTrainer with parameters suitable for Phi-3.

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

        # LoRA parameters
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # Check for hardware acceleration
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info(f"Using Apple Silicon acceleration (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("No GPU detected, using CPU (this might be slower)")

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
        Format the dataset for Phi-3 causal language model training.
        """
        logger.info("Formatting dataset for Phi-3")

        # Identify input and target columns
        input_col, target_col = self._identify_dataset_columns(dataset)
        logger.info(f"Using columns: input={input_col}, target={target_col}")

        def format_prompt(question, answer):
            """Format the prompt and response in Phi-3's expected format."""
            # Extract actual question from the context+question format
            if "Context:" in question and "Question:" in question:
                # Split out just the question part if we're in the RAG format
                parts = question.split("Question:")
                if len(parts) > 1:
                    actual_question = parts[1].strip()
                else:
                    actual_question = question
            else:
                actual_question = question

            # Format in Phi-3's expected chat format
            return f"""<|system|>
You are a first responders chatbot designed to provide accurate information about emergency procedures and protocols based on official training materials.
<|user|>
{actual_question}
<|assistant|>
{answer}"""

        # Check if the dataset already has a 'text' column
        if "text" in dataset.column_names:
            logger.info("Dataset already has a 'text' column, using it directly")
            return dataset

        def format_samples(examples):
            """Process a batch of examples into formatted prompts."""
            inputs = examples[input_col]
            targets = examples[target_col]

            formatted_texts = []
            for q, a in zip(inputs, targets):
                formatted_texts.append(format_prompt(q, a))

            return {"text": formatted_texts}

        # Apply formatting to create prompts in Phi-3 format
        formatted_dataset = dataset.map(
            format_samples,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Formatting prompts",
        )

        # Show a sample of the formatted data
        if len(formatted_dataset) > 0:
            logger.info(
                f"Sample formatted prompt: \n{formatted_dataset[0]['text'][:500]}..."
            )

        return formatted_dataset

    def load_and_prepare_model(self):
        """
        Load and prepare the Phi-3 Mini model with quantization and LoRA.
        """
        logger.info(f"Loading model: {self.model_name}")

        # Configure quantization
        quantization_config = None

        # Check if we're on Apple Silicon (MPS)
        is_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        # Check for CPU only mode
        is_cpu_only = not torch.cuda.is_available() and not is_mps

        # Load model with appropriate settings based on hardware
        if is_mps:
            logger.info("Using Apple Silicon (MPS) with reduced memory footprint")
            # For MPS backend, use lower precision and memory optimization
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                use_cache=False,  # Disable KV cache to save memory during training
            )
            # Enable gradient checkpointing to reduce memory usage
            model.gradient_checkpointing_enable()

            # Further reduce sequence length for faster training
            actual_max_len = min(self.max_seq_length, 512)
            logger.info(
                f"Reducing max sequence length to {actual_max_len} for faster training"
            )
            self.max_seq_length = actual_max_len

        elif self.load_in_4bit and not is_cpu_only:
            logger.info("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            # CPU or other device without quantization
            logger.info("Using standard precision")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16 if self.fp16 else torch.float32,
            )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prepare model for k-bit training if using quantization
        if self.load_in_4bit and not is_mps and not is_cpu_only:
            model = prepare_model_for_kbit_training(model)

        # Configure LoRA with fewer target modules and smaller rank for faster training
        logger.info("Applying optimized LoRA configuration for faster training")

        # Get a list of model modules to find the correct target names
        target_modules = []
        # Search for common attention module names in Phi-3
        for name, _ in model.named_modules():
            if any(pattern in name for pattern in ["attention", "mlp"]):
                logger.info(f"Found potential module: {name}")

        # Use correct module names for Phi-3
        lora_config = LoraConfig(
            r=8,  # Reduced from 16 to 8
            lora_alpha=16,  # Reduced from 32 to 16
            target_modules=[
                "qkv_proj",
                "mlp.gate_proj",
            ],  # Module names that actually exist in Phi-3
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # Log info about trainable parameters

        return model, tokenizer

    def train(
        self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None
    ) -> None:
        """
        Train the Phi-3 model using QLoRA.
        """
        logger.info("Starting model training")

        # Load and prepare model
        model, tokenizer = self.load_and_prepare_model()

        # Format datasets
        formatted_train_dataset = self.format_dataset(train_dataset)
        formatted_eval_dataset = (
            self.format_dataset(eval_dataset) if eval_dataset is not None else None
        )

        # Debug dataset structure
        logger.info(
            f"Formatted train dataset columns: {formatted_train_dataset.column_names}"
        )
        logger.info(f"Sample entry: {formatted_train_dataset[0]}")

        # Tokenize the datasets
        def tokenize_function(examples):
            # Ensure we're working with flat strings, not nested lists
            if "text" in examples:
                texts = examples["text"]
                if isinstance(texts, list) and len(texts) > 0:
                    if isinstance(texts[0], list):
                        logger.warning("Found nested lists in text data, flattening")
                        texts = [
                            (
                                item[0]
                                if isinstance(item, list) and len(item) > 0
                                else item
                            )
                            for item in texts
                        ]

                # Check for Apple Silicon (MPS) to use stricter truncation
                is_mps = (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                )
                max_length = (
                    min(self.max_seq_length, 1024) if is_mps else self.max_seq_length
                )

                # Create a new dictionary with only the tokenized text
                # This avoids any other potentially nested fields
                result = tokenizer(
                    texts,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors=None,  # Return list of integers, not tensors yet
                )

                # If labels are needed for causal LM, copy input_ids to labels
                result["labels"] = result["input_ids"].copy()

                return result
            else:
                raise ValueError(f"Expected 'text' column but found: {examples.keys()}")

        # Process datasets
        logger.info("Tokenizing datasets")
        tokenized_train_dataset = formatted_train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=formatted_train_dataset.column_names,  # Remove all original columns
            desc="Tokenizing training dataset",
        )

        # Log sample tokenized data
        logger.info(
            f"Tokenized train dataset columns: {tokenized_train_dataset.column_names}"
        )
        logger.info(f"Tokenized sample: {tokenized_train_dataset[0]}")

        tokenized_eval_dataset = None
        if formatted_eval_dataset is not None:
            tokenized_eval_dataset = formatted_eval_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=formatted_eval_dataset.column_names,  # Remove all original columns
                desc="Tokenizing evaluation dataset",
            )

        # Create a data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=(
                8 if self.fp16 else None
            ),  # For efficient tensor operations
        )

        # Check for Apple Silicon (MPS)
        is_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        # Use smaller batch size on MPS
        actual_batch_size = 1 if is_mps else self.batch_size

        # Set shorter training for faster completion
        actual_epochs = 1  # Reduce to just 1 epoch

        logger.info(
            f"Using accelerated training settings: {actual_epochs} epochs with batch size {actual_batch_size}"
        )

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=actual_batch_size,
            per_device_eval_batch_size=actual_batch_size,
            gradient_accumulation_steps=(
                8 if is_mps else self.gradient_accumulation_steps
            ),  # Reduced accumulation for speed
            learning_rate=self.learning_rate
            * 2,  # Higher learning rate for faster convergence
            num_train_epochs=actual_epochs,  # Just 1 epoch for faster training
            weight_decay=0.0,  # Disable weight decay for speed
            fp16=self.fp16
            and not is_mps,  # MPS doesn't need this flag as we handle it separately
            warmup_ratio=0.03,  # Shorter warmup
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,  # More frequent logging
            save_strategy="epoch",
            evaluation_strategy="epoch" if tokenized_eval_dataset is not None else "no",
            save_total_limit=1,  # Save less checkpoints
            load_best_model_at_end=tokenized_eval_dataset is not None,
            optim="adamw_torch",  # Use adamw_torch optimizer
            bf16=False,  # bfloat16 precision typically not supported on consumer hardware
            remove_unused_columns=False,  # Important for custom formatting
            gradient_checkpointing=True,  # Enable gradient checkpointing
            dataloader_num_workers=0,  # Avoid DataLoader workers for simpler processing
            group_by_length=True,  # Group sequences of similar length for efficiency
            ddp_find_unused_parameters=False,  # Speed up training
            do_eval=tokenized_eval_dataset is not None,
            no_cuda=True if is_mps else False,  # Ensure we use MPS on Apple Silicon
            report_to="none",  # Skip reporting to save time
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        # Train the model
        logger.info("Training the model")
        trainer.train()

        # Save the model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        # Save LoRA adapter separately for easier loading
        model.save_pretrained(os.path.join(self.output_dir, "adapter"))
        logger.info(f"Saved LoRA adapter to {os.path.join(self.output_dir, 'adapter')}")

    def _identify_dataset_columns(self, dataset: Dataset) -> Tuple[str, str]:
        """
        Identify the input and target columns in the dataset.

        This method attempts to identify the input and target columns in the dataset
        by looking for common column names or patterns.

        Args:
            dataset: The dataset to analyze

        Returns:
            tuple: (input_column_name, target_column_name)
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

        # Train the model
        self.train(train_dataset=train_dataset, eval_dataset=eval_dataset)
