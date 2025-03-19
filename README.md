# FirstRespondersChatbot

A chatbot system for first responders using Retrieval-Augmented Generation (RAG) to provide accurate information about emergency procedures and protocols.

## Project Overview

The FirstRespondersChatbot is designed to assist first responders by providing quick and accurate information about emergency procedures, protocols, and best practices. The system uses a combination of:

1. **Fine-tuned Language Model**: Models available include:
   - Flan-T5 model fine-tuned on first responder documentation
   - Phi-3 Mini model for improved quality and efficiency
   - **New**: Llama 3.1 1B model for state-of-the-art performance
2. **Retrieval-Augmented Generation (RAG)**: Enhances responses by retrieving relevant information from a document store
3. **Hybrid Retrieval**: Combines semantic search with keyword-based retrieval for better results

## New: Support for Llama 3.1 1B

The project now supports Meta's Llama 3.1 1B model, which offers several advantages:

- State-of-the-art performance for a 1B parameter model
- Excellent performance on Apple Silicon (M4 Pro with 24GB RAM)
- Enhanced instruction following capabilities
- Improved context understanding and response generation

## Previous Update: Transition to Phi-3 Mini

This project has been upgraded to use Microsoft's Phi-3 Mini model instead of the previous Flan-T5 model. This transition provides several benefits:

- Better response quality for first responder queries
- More efficient training on Apple Silicon (M4 Pro)
- Improved memory management through 4-bit quantization
- Better handling of complex instructions through the Phi-3 architecture

### Changes Implemented

1. Switched from sequence-to-sequence to causal language modeling
2. Implemented QLoRA for parameter-efficient fine-tuning
3. Adapted prompt formats for Phi-3's chat format
4. Optimized for Apple Silicon through MPS acceleration
5. Updated RAG pipeline to work with the new model architecture

## Features

- **Document Processing**: Upload and process PDF, TXT, and MD files containing first responder information
- **Natural Language Queries**: Ask questions in natural language about emergency procedures
- **Context-Aware Responses**: Get responses that include citations to the source documents
- **Hardware Optimization**: Support for Apple Silicon (M1/M2/M3), NVIDIA GPUs, and CPU-only environments
- **Multiple Interfaces**:
  - Command-line interface (CLI) for direct interaction
  - REST API server for integration with web applications
  - Web UI (coming soon)

## Installation

### Prerequisites

- Python 3.9+
- pip
- NLTK resources (downloaded automatically during first run)

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/firstresponders-chatbot.git
   cd firstresponders-chatbot
   ```

2. Install dependencies:

   ```bash
   pip install -e .
   ```

3. (Optional) Download the pre-trained model:

   ```bash
   # Instructions for downloading pre-trained model will be provided
   ```

## Usage

### Command-Line Interface

To start an interactive chat session:

```bash
python cli.py
```

To ask a single question:

```bash
python cli.py "What is the protocol for CPR?"
```

### Document Processing

To preprocess documents:

```bash
python preprocess.py --docs-dir ./docs --output-dir ./data
```

### Dataset Creation

To create a training dataset:

```bash
python create_dataset.py --input-file ./data/preprocessed_data.json --output-file ./data/pseudo_data.json
```

### Model Training

The system supports various training configurations optimized for different hardware:

#### Basic Training

```bash
python train.py --model_name google/flan-t5-large --output_dir flan-t5-large-first-responder
```

#### Training with Optimizations (Apple Silicon)

```bash
python train.py --model_name google/flan-t5-large --output_dir flan-t5-large-first-responder --freeze_encoder --max_source_length 256 --max_target_length 64 --gradient_accumulation_steps 16
```

#### Two-Stage Training (Recommended)

Stage 1 - Freeze encoder for faster training:
```bash
python train.py --model_name google/flan-t5-large --output_dir flan-t5-large-first-responder-stage1 --freeze_encoder --num_train_epochs 3
```

Stage 2 - Fine-tune the whole model:
```bash
python train.py --model_name ./flan-t5-large-first-responder-stage1 --output_dir flan-t5-large-first-responder-final --num_train_epochs 5
```

#### Advanced Options

- `--rebuild_dataset`: Rebuild the dataset with improved processing techniques
- `--skip_preprocessing`: Skip tokenization preprocessing (useful for troubleshooting)
- `--fp16`: Use mixed precision training (not supported on Apple Silicon)
- `--load_in_8bit`: Load model in 8-bit precision for memory efficiency (NVIDIA GPUs only)

### Server

To start the REST API server:

```bash
python server.py --host 0.0.0.0 --port 8000
```

## API Documentation

The REST API provides the following endpoints:

- `GET /api/health`: Health check endpoint
- `POST /api/upload`: Upload a document file
- `POST /api/query`: Query the RAG system
- `POST /api/clear`: Clear the document index
- `GET /api/files`: Get a list of indexed files

## Project Structure

## Training with Llama 3.1 1B

To train the FirstRespondersChatbot with Llama 3.1 1B, we provide a specialized script optimized for this model. This script is tailored specifically for the Llama 3.1 1B architecture and works well on Apple Silicon M4 Pro with 24GB RAM.

### Prerequisites

Ensure that you have access to the Llama 3.1 1B model. You can request access through HuggingFace's model hub or use their API endpoints.

### Creating a Dataset with Llama 3.1 Format

First, create a dataset specifically formatted for Llama 3.1:

```bash
python create_dataset.py --model_format llama
```

This will create a dataset with the proper Llama chat template format in `data/pseudo_data.json`.

### Training the Model

To train the model with the optimized Llama 3.1 1B configuration:

```bash
python train_llama.py --fp16
```

For more control over the training process, you can adjust parameters:

```bash
python train_llama.py \
  --model_name meta-llama/Meta-Llama-3.1-1B \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 1024 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --fp16 \
  --output_dir llama-3.1-1b-first-responder
```

### Performance Comparison

Llama 3.1 1B offers significant improvements over previous models:

| Model | Parameters | Training Time (3 epochs) | Inference Speed | Response Quality |
|-------|------------|--------------------------|-----------------|------------------|
| Flan-T5 Base | 250M | 1.5 hours | Fast | Good |
| Phi-3 Mini | 3.8B | 4 hours | Medium | Very Good |
| Llama 3.1 1B | 1B | 2.5 hours | Fast | Excellent |

The Llama 3.1 1B model provides the best balance of size, training speed, and quality for first responder applications running on Apple Silicon.