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
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dataset_path: str = "data/pseudo_data.json",
        output_dir: str = "tinyllama-1.1b-first-responder-fast",
        batch_size: int = 1,
        learning_rate: float = 2e-4,
        num_train_epochs: int = 3,
        max_seq_length: int = 512,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 16,
        fp16: bool = True,
        load_in_4bit: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        max_train_samples: Optional[
            int
        ] = None,  # Added parameter for limiting training data
        use_8bit_optimizer: bool = True,  # Control 8-bit optimizer usage
    ):
        """
        Initialize the ModelTrainer with parameters suitable for TinyLlama or Llama.

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
        Format the dataset for TinyLlama causal language model training.
        """
        logger.info("Formatting dataset for TinyLlama")

        # Identify input and target columns
        input_col, target_col = self._identify_dataset_columns(dataset)
        logger.info(f"Using columns: input={input_col}, target={target_col}")

        def format_prompt(question, answer):
            """Format the prompt and response in TinyLlama's expected format."""
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

            # Format in TinyLlama's expected chat format
            return f"""<s>[INST] <<SYS>>
You are a first responders chatbot designed to provide accurate information about emergency procedures and protocols based on official training materials.
<</SYS>>

{actual_question}
[/INST]
{answer}</s>"""

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

        # Apply formatting to create prompts in TinyLlama format
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

    def _load_and_prepare_model(self):
        """
        Load and prepare the TinyLlama model with quantization and LoRA.
        """
        logger.info(f"Loading model {self.model_name} for training...")

        # Configure quantization
        quantization_config = None

        # Check if we're on Apple Silicon (MPS)
        is_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        # Check for CPU only mode
        is_cpu_only = not torch.cuda.is_available() and not is_mps

        # Load model with appropriate settings based on hardware
        if is_mps:
            logger.info(
                "Using Apple Silicon (MPS) - disabling quantization for compatibility"
            )
            quantization_config = None
            self.load_in_4bit = False
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

        # Simplify by using known working target modules for TinyLlama specifically
        if "TinyLlama" in self.model_name:
            logger.info("Detected TinyLlama model, using known working target modules")
            # According to TinyLlama model architecture, these are the correct modules
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            # For other models, use a general approach
            if hasattr(model, "get_decoder"):
                target_modules = find_all_linear_names(model.get_decoder())
            else:
                target_modules = find_all_linear_names(model)

        logger.info(f"Using target modules: {target_modules}")

        # Simplified LoRA config
        lora_config = LoraConfig(
            r=8,  # Reduced rank for faster training
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=["lm_head"],  # Save the LM head for generation
        )

        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # Log info about trainable parameters

        return model, tokenizer

    def train(
        self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None
    ) -> None:
        """Train the model on the given datasets.

        Args:
            train_dataset: Dataset to train on
            eval_dataset: Optional dataset to evaluate on during training
        """
        # Log start of training process
        logger.info("Starting model training")
        logger.info("Loading and preparing model...")

        # Load the pre-trained model
        logger.info(f"Loading model: {self.model_name}")

        # Check if Apple Silicon is available
        is_apple_silicon = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

        # Set device based on availability
        if is_apple_silicon:
            logger.info("Apple Silicon (M4 Pro) detected - using MPS")
            device_choice = "mps"
            # Disable quantization for M-series chip training
            logger.info("Adapting configuration for Apple Silicon")
            quantization_config = None
            self.load_in_4bit = False
            use_8bit = False
        elif torch.cuda.is_available():
            device_choice = "cuda"
            # Set up quantization for CUDA if requested
            use_8bit = self.use_8bit_optimizer
            if self.load_in_4bit and BITSANDBYTES_AVAILABLE:
                logger.info("Using 4-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                quantization_config = None
        else:
            logger.info("No GPU detected, using CPU (this might be slower)")
            device_choice = "cpu"
            quantization_config = None
            self.load_in_4bit = False
            use_8bit = False

        # Check if we're using Llama model
        is_llama_model = "llama" in self.model_name.lower()

        # Set torch dtype based on model and hardware
        if is_apple_silicon:
            # Use float16 for efficient MPS training on Apple Silicon
            model_dtype = torch.float16 if self.fp16 else torch.float32
        else:
            # Use bfloat16 for Llama on other hardware as it works better
            model_dtype = (
                torch.bfloat16
                if is_llama_model and self.fp16
                else torch.float16 if self.fp16 else torch.float32
            )

        # Load model with appropriate configuration
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            device_map=(
                device_choice if device_choice != "mps" else "auto"
            ),  # mps needs special handling
        )

        # Move model to MPS device if using Apple Silicon
        if device_choice == "mps":
            model = model.to("mps")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Configure appropriate LoRA settings based on model architecture
        if is_llama_model:
            # Target attention modules for Llama
            logger.info("Configuring LoRA for Llama model")
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            # Target attention modules for TinyLlama (and others)
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            logger.info(f"Using target modules: {target_modules}")

        # Apply PEFT configuration
        if not self.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        model = get_peft_model(model, peft_config)

        logger.info(f"Model has {model.num_parameters()} parameters")
        model.print_trainable_parameters()

        # Format datasets based on model type
        logger.info("Formatting datasets for training")

        # Check and identify input/target columns
        input_col, target_col = self._identify_dataset_columns(train_dataset)
        logger.info(
            f"Using '{input_col}' as input column and '{target_col}' as target column"
        )

        # Handle dataset formatting for different model types
        if is_llama_model:
            logger.info("Formatting dataset for Llama 3.1")
            formatted_train_dataset = self._format_dataset_for_llama(
                train_dataset, input_col, target_col
            )
            formatted_eval_dataset = None
            if eval_dataset is not None:
                formatted_eval_dataset = self._format_dataset_for_llama(
                    eval_dataset, input_col, target_col
                )
        else:
            # Use existing formatting for TinyLlama or other models
            formatted_train_dataset = self._format_dataset_for_tinyllama(
                train_dataset, input_col, target_col
            )
            formatted_eval_dataset = None
            if eval_dataset is not None:
                formatted_eval_dataset = self._format_dataset_for_tinyllama(
                    eval_dataset, input_col, target_col
                )

        # Tokenization function
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

                # Adjust max_length based on available memory and hardware
                if is_apple_silicon:
                    # M4 Pro with 24GB can handle larger sequences
                    max_length = min(self.max_seq_length, 1024)
                else:
                    max_length = min(self.max_seq_length, 512)

                logger.info(f"Using max_length={max_length} for tokenization")

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
        logger.info("Tokenizing train dataset...")
        tokenized_train_dataset = formatted_train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=formatted_train_dataset.column_names,  # Remove all original columns
            desc="Tokenizing training dataset",
        )
        logger.info("Train dataset tokenized successfully")

        # Log sample tokenized data
        logger.info(
            f"Tokenized train dataset columns: {tokenized_train_dataset.column_names}"
        )
        logger.info(f"Tokenized sample: {tokenized_train_dataset[0]}")

        tokenized_eval_dataset = None
        if formatted_eval_dataset is not None:
            logger.info("Tokenizing eval dataset...")
            tokenized_eval_dataset = formatted_eval_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=formatted_eval_dataset.column_names,  # Remove all original columns
                desc="Tokenizing evaluation dataset",
            )
            logger.info("Eval dataset tokenized successfully")

        # Create a data collator for language modeling
        logger.info("Creating data collator...")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        logger.info("Data collator created")

        # Adjust batch size based on model and hardware
        if is_apple_silicon:
            # M4 Pro with 24GB can handle a bit larger batch size
            actual_batch_size = 2 if is_llama_model else 1
        else:
            actual_batch_size = 1

        # Set training epochs
        actual_epochs = self.num_train_epochs

        logger.info(
            f"Using training settings: {actual_epochs} epochs with batch size {actual_batch_size}"
        )

        # Set up the training arguments
        logger.info("Setting up training arguments...")

        # Use appropriate optimizer based on hardware
        if is_apple_silicon:
            logger.info("Using standard AdamW optimizer for Apple Silicon")
            optim = "adamw_torch"
        else:
            if use_8bit and BITSANDBYTES_AVAILABLE:
                logger.info("Using 8-bit AdamW optimizer")
                optim = "adamw_8bit"
            else:
                logger.info("Using standard AdamW optimizer")
                optim = "adamw_torch"

        # Configure training args
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=actual_epochs,
            per_device_train_batch_size=actual_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            logging_steps=1,
            logging_dir=os.path.join(self.output_dir, "logs"),
            save_total_limit=2,
            load_best_model_at_end=eval_dataset is not None,
            fp16=self.fp16 and device_choice != "cpu",
            optim=optim,
            # Configure for appropriate device
            use_cpu=device_choice == "cpu",
            no_cuda=device_choice != "cuda",
        )
        logger.info("Training arguments set up")

        # Create trainer
        logger.info("Creating trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        logger.info("Trainer created successfully")

        # Train the model
        logger.info("Starting training loop...")
        try:
            logger.info("Calling trainer.train()")
            trainer.train()
            logger.info("Training completed successfully!")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            import traceback

            logger.error(traceback.format_exc())
            raise

        # Save the model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        # Save LoRA adapter separately for easier loading
        model.save_pretrained(os.path.join(self.output_dir, "adapter"))

        logger.info("Model and tokenizer saved successfully")

    def _format_dataset_for_tinyllama(self, dataset, input_col, target_col):
        """Format a dataset for TinyLlama training"""
        system_message = "You are a first responders chatbot designed to provide accurate information about emergency procedures and protocols based on official training materials."

        def format_example(example):
            question = example[input_col]
            answer = example[target_col]
            return {
                "text": f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{question} [/INST] {answer}</s>"
            }

        return dataset.map(format_example)

    def _format_dataset_for_llama(self, dataset, input_col, target_col):
        """Format a dataset for Llama 3.1 training"""
        system_message = "You are a first responders chatbot designed to provide accurate information about emergency procedures and protocols based on official training materials."

        def format_example(example):
            question = example[input_col]
            answer = example[target_col]
            return {
                "text": f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{question} [/INST] {answer}</s>"
            }

        return dataset.map(format_example)

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
