# FirstRespondersChatbot

A chatbot system for first responders using Retrieval-Augmented Generation (RAG) to provide accurate information about emergency procedures and protocols.

## Project Overview

The FirstRespondersChatbot is designed to assist first responders by providing quick and accurate information about emergency procedures, protocols, and best practices. The system uses a combination of:

1. **Fine-tuned Language Model**: A Flan-T5 model fine-tuned on first responder documentation
2. **Retrieval-Augmented Generation (RAG)**: Enhances responses by retrieving relevant information from a document store
3. **Hybrid Retrieval**: Combines semantic search with keyword-based retrieval for better results

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

```
firstresponders-chatbot/
├── src/
│   └── firstresponders_chatbot/
│       ├── cli/              # Command-line interface
│       ├── preprocessing/    # Document preprocessing
│       ├── rag/              # RAG system with hybrid retrieval
│       └── training/         # Model training with hardware optimizations
├── data/                     # Data storage
├── docs/                     # Documentation and example files
├── uploads/                  # Uploaded files
├── cli.py                    # CLI entry point
├── server.py                 # Server entry point
├── preprocess.py             # Preprocessing entry point
├── create_dataset.py         # Dataset creation entry point
├── train.py                  # Training entry point
├── main.py                   # Main entry point
└── pyproject.toml            # Dependencies
```

## Recent Improvements

- **Enhanced Document Processing**: Better text cleaning and deduplication
- **Hardware Optimization**: Support for Apple Silicon (MPS), NVIDIA GPUs, and efficient CPU operation
- **Memory Efficiency**: Dynamic sequence length handling, gradient accumulation, and optional quantization
- **Training Pipeline**: Two-stage training approach for better results with less training time
- **Evaluation Metrics**: Added ROUGE and BLEU score calculations for model evaluation
- **Error Handling**: Improved error detection and recovery during training
- **Haystack 2.0 Integration**: Updated to use the latest Haystack API for better retrieval quality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the Transformers library
- Haystack for the RAG components
- All contributors to the project
