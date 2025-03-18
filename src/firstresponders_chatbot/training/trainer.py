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
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset, load_dataset
import evaluate  # Import evaluate package for metrics
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
        model_name: str = "google/flan-t5-small",
        dataset_path: str = "data/pseudo_data.json",
        output_dir: str = "flan-t5-first-responder",
        batch_size: int = 1,  # Reduced batch size for larger models
        learning_rate: float = 3e-5,  # Slightly lower learning rate
        num_train_epochs: int = 8,  # More epochs for better learning
        max_source_length: int = 384,
        max_target_length: int = 96,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 32,  # Increased for memory efficiency
        fp16: bool = True,
        freeze_encoder: bool = False,  # Option to freeze encoder layers
        load_in_8bit: bool = True,  # Enable 8-bit quantization
    ):
        """
        Initialize the ModelTrainer.

        Args:
            model_name: Name of the pre-trained model to use
            dataset_path: Path to the training dataset
            output_dir: Directory to save the trained model
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            num_train_epochs: Number of epochs to train for
            max_source_length: Maximum length of the source sequences
            max_target_length: Maximum length of the target sequences
            weight_decay: Weight decay for regularization
            warmup_ratio: Ratio of total training steps used for learning rate warmup
            gradient_accumulation_steps: Number of steps to accumulate gradients before performing an update
            fp16: Whether to use mixed precision training
            freeze_encoder: Whether to freeze the encoder layers
            load_in_8bit: Whether to load model in 8-bit precision
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.freeze_encoder = freeze_encoder
        self.load_in_8bit = load_in_8bit

        # Check if CUDA is available
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

            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def preprocess_data(self, dataset: Dataset) -> Dataset:
        """
        Preprocess the dataset for training.

        Args:
            dataset: The dataset to preprocess

        Returns:
            The preprocessed dataset
        """
        logger.info("Preprocessing dataset")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Identify input and target columns
        input_col, target_col = self._identify_dataset_columns(dataset)
        logger.info(
            f"Using columns for preprocessing: input={input_col}, target={target_col}"
        )

        # Log dataset sample for debugging
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(
                f"Sample example - {input_col}: {sample[input_col][:100]}..., {target_col}: {sample[target_col][:100]}..."
            )

        def preprocess_function(examples):
            # Extract inputs and targets directly from identified columns
            inputs = examples[input_col]
            targets = examples[target_col]

            # Log length statistics
            if isinstance(inputs, list) and len(inputs) > 0:
                input_lengths = [len(inp.split()) for inp in inputs]
                target_lengths = [len(tgt.split()) for tgt in targets]
                logger.info(
                    f"Input length stats: min={min(input_lengths)}, max={max(input_lengths)}, avg={sum(input_lengths)/len(input_lengths):.1f}"
                )
                logger.info(
                    f"Target length stats: min={min(target_lengths)}, max={max(target_lengths)}, avg={sum(target_lengths)/len(target_lengths):.1f}"
                )

            # Tokenize inputs
            model_inputs = tokenizer(
                inputs,
                max_length=self.max_source_length,
                padding="max_length",
                truncation=True,
            )

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=self.max_target_length,
                    padding="max_length",
                    truncation=True,
                )

            # Replace padding token id with -100 in labels for loss masking
            labels_with_ignore = []
            for label in labels["input_ids"]:
                # Replace padding token id with -100
                label_with_ignore = [
                    -100 if token == tokenizer.pad_token_id else token
                    for token in label
                ]
                labels_with_ignore.append(label_with_ignore)

            model_inputs["labels"] = labels_with_ignore
            return model_inputs

        # Process the dataset - crucially this removes all original columns and
        # replaces them with tokenized versions
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,  # Critical: remove original columns
            desc="Tokenizing dataset",
            num_proc=1,  # Single process to avoid conflicts
        )

        # Log processed dataset info
        logger.info(f"Processed dataset columns: {processed_dataset.column_names}")
        logger.info(f"Processed dataset size: {len(processed_dataset)}")

        return processed_dataset

    def train(
        self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None
    ) -> None:
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
        """
        logger.info("Starting model training")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Verify dataset format is correct
        logger.info(f"Training dataset columns: {train_dataset.column_names}")
        expected_columns = ["input_ids", "attention_mask", "labels"]

        # Only convert if needed - prefer using preprocess_data first
        missing_columns = [
            col for col in expected_columns if col not in train_dataset.column_names
        ]
        if missing_columns:
            logger.warning(f"Dataset missing expected columns: {missing_columns}")
            logger.info("Converting dataset format automatically")
            train_dataset = self._convert_dataset_format(train_dataset, tokenizer)
            if eval_dataset is not None:
                eval_dataset = self._convert_dataset_format(eval_dataset, tokenizer)
        else:
            logger.info(
                "Dataset already in the correct format with input_ids and labels"
            )

        # Quantization config for memory efficiency
        quantization_config = None
        if (
            self.load_in_8bit
            and "mps" not in str(self.device)
            and BITSANDBYTES_AVAILABLE
        ):  # 8-bit not fully supported on MPS
            logger.info("Loading model in 8-bit precision")
            quantization_config = {"load_in_8bit": True}
        elif self.load_in_8bit:
            logger.warning(
                "8-bit quantization requested but not available - continuing with standard precision"
            )

        # Load model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, **(quantization_config if quantization_config else {})
        )

        # Configure memory optimization settings
        use_mps = torch.backends.mps.is_available()

        # Create training arguments
        gradient_checkpointing = False
        if not self.freeze_encoder:
            # Only use gradient checkpointing for full model training
            gradient_checkpointing = True
            logger.info("Enabling gradient checkpointing")
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
        else:
            logger.info("Gradient checkpointing disabled (using frozen encoder)")

        # Freeze encoder if requested
        if self.freeze_encoder:
            logger.info("Freezing encoder layers")
            for param in model.encoder.parameters():
                param.requires_grad = False

        # Move model to device
        model = model.to(self.device)

        # Create data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding="longest",
        )

        # Load metrics
        rouge_metric = evaluate.load("rouge")
        bleu_metric = evaluate.load("bleu")

        # Define compute metrics function
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]

            # Replace -100 with pad token id
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            # Decode predictions and labels
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Normalize whitespace
            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]

            # Initialize metrics
            rouge_scorer = evaluate.load("rouge")

            # Compute ROUGE
            result = rouge_scorer.compute(
                predictions=decoded_preds, references=decoded_labels, use_stemmer=True
            )

            # Extract ROUGE scores - handle both old and new ROUGE implementation
            results = {}
            for k, v in result.items():
                # Handle different versions of ROUGE metrics
                if hasattr(v, "mid"):
                    # Old version
                    results[k] = v.mid.fmeasure * 100
                elif isinstance(v, float):
                    # New version - directly returns float values
                    results[k] = v * 100
                else:
                    # Unexpected type
                    results[k] = 0.0
                    logger.warning(f"Unexpected type for ROUGE metric {k}: {type(v)}")

            # Add prediction length
            prediction_lens = [len(pred.split()) for pred in decoded_preds]
            results["gen_len"] = np.mean(prediction_lens)

            return results

        # Define training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            num_train_epochs=self.num_train_epochs,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            fp16=self.fp16
            and "mps" not in str(self.device),  # fp16 not fully supported on MPS
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=50,
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset is not None else "no",
            save_total_limit=2,
            predict_with_generate=True,
            generation_max_length=self.max_target_length,
            generation_num_beams=4,
            load_best_model_at_end=True if eval_dataset is not None else False,
            metric_for_best_model="rouge1" if eval_dataset is not None else None,
            greater_is_better=True,
            remove_unused_columns=False,  # Important to prevent column mismatch errors
        )

        # Create trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics if eval_dataset is not None else None,
        )

        # Train the model
        logger.info("Training the model")
        trainer.train()

        # Save the model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

    def _convert_dataset_format(self, dataset: Dataset, tokenizer) -> Dataset:
        """
        Convert a dataset with columns like 'input'/'output' to the expected format with 'input_ids'/'labels'.

        Args:
            dataset: The dataset to convert
            tokenizer: The tokenizer to use for conversion

        Returns:
            Converted dataset with proper format
        """
        logger.info(f"Converting dataset with columns: {dataset.column_names}")

        # Identify input and target columns
        input_col, target_col = self._identify_dataset_columns(dataset)
        logger.info(f"Identified columns: input={input_col}, target={target_col}")

        def convert_function(examples):
            # Extract inputs and targets
            inputs = examples[input_col]
            targets = examples[target_col]

            # Tokenize inputs
            model_inputs = tokenizer(
                inputs,
                max_length=self.max_source_length,
                padding="max_length",
                truncation=True,
            )

            # Tokenize targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=self.max_target_length,
                    padding="max_length",
                    truncation=True,
                )

            # Replace padding token id with -100 in labels
            labels_with_ignore = []
            for label in labels["input_ids"]:
                label_with_ignore = [
                    -100 if token == tokenizer.pad_token_id else token
                    for token in label
                ]
                labels_with_ignore.append(label_with_ignore)

            model_inputs["labels"] = labels_with_ignore
            return model_inputs

        # Apply conversion
        converted_dataset = dataset.map(
            convert_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Converting dataset format",
            num_proc=1,  # Use single process to avoid potential conflicts
        )

        logger.info(f"Converted dataset columns: {converted_dataset.column_names}")
        return converted_dataset

    def _identify_dataset_columns(self, dataset: Dataset) -> Tuple[str, str]:
        """
        Identify the input and target columns in the dataset.

        Args:
            dataset: The dataset to analyze

        Returns:
            Tuple of (input_column, target_column)
        """
        columns = dataset.column_names
        logger.info(f"Dataset column names: {columns}")

        # Log a sample to help with debugging
        if len(dataset) > 0:
            logger.info(f"Dataset sample: {dataset[0]}")

        # Check for standard column patterns
        if "question" in columns and "answer" in columns:
            return "question", "answer"
        elif "input" in columns and "output" in columns:
            return "input", "output"
        elif "source" in columns and "target" in columns:
            return "source", "target"
        elif "prompt" in columns and "completion" in columns:
            return "prompt", "completion"
        elif "text" in columns and "labels" in columns:
            return "text", "labels"

        # Default to first two columns
        if len(columns) >= 2:
            logger.warning(
                f"Using first two columns as input/output: {columns[0]}, {columns[1]}"
            )
            return columns[0], columns[1]
        elif len(columns) == 1:
            logger.warning(
                f"Only one column found ({columns[0]}), using it for both input and output"
            )
            return columns[0], columns[0]
        else:
            raise ValueError("Dataset has no columns")

    def run(self) -> None:
        """Run the complete training pipeline."""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Load and preprocess the dataset
        dataset = self.load_dataset()
        processed_dataset = self.preprocess_data(dataset)

        # Train the model
        self.train(processed_dataset)

        logger.info("Training completed successfully!")
