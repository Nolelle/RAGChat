#!/usr/bin/env python3
"""
train.py - Model training script for the FirstRespondersChatbot project.

This script fine-tunes the Flan-T5-Small model using the dataset generated
by create_dataset.py. It supports training on either NVIDIA GPU (Windows)
or Apple Silicon (MacBook Pro) and saves the fine-tuned model to the
flan-t5-first-responder/ directory.
"""

import json
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the Hugging Face cache directory to D drive
os.environ["TRANSFORMERS_CACHE"] = "D:/SAIT/winter_2025/emergin_trends/RAGChat/hf_cache"
os.environ["HF_HOME"] = "D:/SAIT/winter_2025/emergin_trends/RAGChat/hf_cache"
logger.info(f"Setting Hugging Face cache to: {os.environ['TRANSFORMERS_CACHE']}")

# File paths
DATA_DIR = Path("data")
DATASET_PATH = DATA_DIR / "pseudo_data.json"
MODEL_DIR = Path("flan-t5-first-responder")

# Model configuration
MODEL_NAME = "google/flan-t5-small"  # Small model for faster training
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 3e-5


def detect_hardware():
    """
    Detect available hardware for training.

    Returns:
        tuple: (device_name, device)
    """
    # Check if CUDA is available (NVIDIA GPU)
    if torch.cuda.is_available():
        device_name = f"CUDA (NVIDIA {torch.cuda.get_device_name(0)})"
        device = torch.device("cuda")
        logger.info(f"Using GPU: {device_name}")

    # Check if MPS is available (Apple Silicon)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = f"MPS (Apple Silicon)"
        device = torch.device("mps")
        logger.info(f"Using {device_name}")

    # Otherwise, use CPU
    else:
        device_name = f"CPU ({platform.processor()})"
        device = torch.device("cpu")
        logger.info(f"No GPU detected, using {device_name}")

    return device_name, device


def load_dataset() -> Dataset:
    """
    Load dataset from JSON file and convert to HuggingFace Dataset.

    Returns:
        Dataset: HuggingFace Dataset object.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    with open(DATASET_PATH, "r") as f:
        data = json.load(f)

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict(
        {
            "input": [item["input"] for item in data],
            "target": [item["target"] for item in data],
        }
    )

    logger.info(f"Loaded dataset with {len(dataset)} examples")

    # Split dataset into train and validation sets (90% / 10%)
    split_dataset = dataset.train_test_split(test_size=0.1)

    logger.info(f"Training set: {len(dataset['train'])} examples")
    logger.info(f"Validation set: {len(dataset['test'])} examples")

    logger.info(f"Training set: {len(split_dataset['train'])} examples")
    logger.info(f"Validation set: {len(split_dataset['validation'])} examples")

    return split_dataset


def preprocess_function(examples, tokenizer):
    """Preprocess the examples by tokenizing."""
    inputs = [doc for doc in examples["input"]]
    targets = [doc for doc in examples["target"]]

    model_inputs = tokenizer(
        inputs, max_length=MAX_INPUT_LENGTH, padding="max_length", truncation=True
    )

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=MAX_TARGET_LENGTH, padding="max_length", truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train_model(dataset, device):
    """
    Fine-tune the Flan-T5-Small model on the dataset.

    Args:
        dataset: HuggingFace Dataset object.
        device: PyTorch device to use for training.

    Returns:
        tuple: (trained model, tokenizer)
    """
    # Create cache directory if it doesn't exist
    cache_dir = os.environ["TRANSFORMERS_CACHE"]
    os.makedirs(cache_dir, exist_ok=True)

    # Load model and tokenizer
    logger.info(f"Loading model {MODEL_NAME}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)

    # Move model to device
    model = model.to(device)

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, return_tensors="pt"
    )

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=3,  # Only keep the 3 most recent checkpoints
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        fp16=device.type == "cuda",  # Use fp16 if on GPU
        report_to="none",  # Disable wandb and other reporting
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training
    logger.info("Starting model training...")
    trainer.train()

    # Save model and tokenizer
    logger.info(f"Saving model to {MODEL_DIR}...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    return model, tokenizer


def main():
    """Main function to train the model."""
    logger.info("Starting Flan-T5-Small fine-tuning process...")

    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Detect hardware
    device_name, device = detect_hardware()
    logger.info(f"Training on: {device_name}")

    # Load dataset
    dataset = load_dataset()

    # Train model
    model, tokenizer = train_model(dataset, device)

    logger.info("Model training completed successfully!")

    # Test the model with a sample question
    test_question = "question: What is the protocol for CPR?"
    input_ids = tokenizer(test_question, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(
        input_ids,
        max_length=100,
        num_beams=4,  # Use beam search for better quality
        temperature=0.7,  # Add some randomness but not too much
        no_repeat_ngram_size=2,  # Avoid repetition
        early_stopping=True,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    logger.info("----- Sample Model Output -----")
    logger.info(f"Question: {test_question}")
    logger.info(f"Answer: {answer}")
    logger.info("------------------------------")


if __name__ == "__main__":
    main()
