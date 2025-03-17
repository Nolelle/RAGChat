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
        batch_size: int = 2,
        learning_rate: float = 5e-5,
        num_train_epochs: int = 5,
        max_source_length: int = 384,
        max_target_length: int = 96,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 16,
        fp16: bool = True,
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

        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

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

        def preprocess_function(examples):
            # Check for different possible column structures
            if "input" in examples and "output" in examples:
                # Format from DatasetCreator
                inputs = examples["input"]
                targets = examples["output"]
            elif "question" in examples and "answer" in examples:
                # Original expected format
                inputs = examples["question"]
                targets = examples["answer"]
            elif "content" in examples:
                # Format from preprocessed_data.json
                inputs = examples["content"]
                targets = examples["content"]
            else:
                # Fallback to using whatever columns are available
                logger.warning(f"Unexpected column structure: {examples.keys()}")
                inputs = examples[dataset.column_names[0]]
                targets = examples[dataset.column_names[0]]

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

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # Process the dataset
        processed_dataset = dataset.map(
            preprocess_function, batched=True, remove_columns=dataset.column_names
        )

        return processed_dataset

    def train(self, dataset: Dataset) -> None:
        """
        Train the model on the provided dataset.

        Args:
            dataset: The dataset to train on
        """
        logger.info(f"Starting training with model {self.model_name}")

        # Split dataset into training and evaluation sets (90% train, 10% eval)
        dataset = dataset.shuffle(
            seed=42
        )  # Shuffle the dataset with a fixed seed for reproducibility
        train_test_split = dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]

        logger.info(
            f"Split dataset into {len(train_dataset)} training examples and {len(eval_dataset)} evaluation examples"
        )

        # Load model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load ROUGE metric for evaluation
        rouge_metric = evaluate.load("rouge")

        # Define compute metrics function
        def compute_metrics(eval_preds):
            preds, labels = eval_preds

            try:
                # Filter out extremely large token IDs that might cause overflow
                max_token_id = tokenizer.vocab_size

                # Ensure preds doesn't have token IDs out of range
                if isinstance(preds, np.ndarray):
                    preds = np.where(
                        preds < max_token_id, preds, tokenizer.pad_token_id
                    )
                else:
                    # If it's a tensor
                    preds = torch.where(
                        preds < max_token_id,
                        preds,
                        torch.tensor(tokenizer.pad_token_id, device=preds.device),
                    )

                # Decode generated summaries
                decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

                # Replace -100 in the labels as we can't decode them
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                # Also filter large token IDs in labels
                labels = np.where(labels < max_token_id, labels, tokenizer.pad_token_id)
                decoded_labels = tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )

                # ROUGE expects newlines after each sentence
                decoded_preds = ["\n".join(pred.split()) for pred in decoded_preds]
                decoded_labels = ["\n".join(label.split()) for label in decoded_labels]

                # Compute ROUGE scores
                result = rouge_metric.compute(
                    predictions=decoded_preds,
                    references=decoded_labels,
                    use_stemmer=True,
                )

                # Extract ROUGE f1 scores
                result = {
                    key: value.mid.fmeasure * 100 for key, value in result.items()
                }

                # Add mean generated length
                prediction_lens = [len(pred.split()) for pred in decoded_preds]
                result["gen_len"] = np.mean(prediction_lens)

                return {k: round(v, 4) for k, v in result.items()}

            except OverflowError as e:
                logger.error(f"OverflowError in compute_metrics: {e}")
                # Return a default metric value if decoding fails
                return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "gen_len": 0.0}
            except Exception as e:
                logger.error(f"Error in compute_metrics: {e}")
                # Return a default metric value if something else goes wrong
                return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "gen_len": 0.0}

        # Check if fp16 is supported on the current device
        use_fp16 = self.fp16
        if self.device.type != "cuda":
            logger.warning(
                f"fp16 is not supported on {self.device.type} devices. Disabling fp16."
            )
            use_fp16 = False

        # Define training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            save_total_limit=2,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            fp16=use_fp16,
            predict_with_generate=True,
            generation_max_length=self.max_target_length,
            report_to="tensorboard",
            # Memory optimization
            dataloader_pin_memory=False,  # Disable pinned memory to reduce memory usage
            optim="adamw_torch",  # Use the PyTorch optimizer implementation
        )

        # Create data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding="max_length",
            max_length=self.max_source_length,
        )

        # Create trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,  # Add evaluation dataset
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,  # Add compute_metrics function
        )

        # Train the model
        logger.info("Training the model")
        trainer.train()

        # Save the model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

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
