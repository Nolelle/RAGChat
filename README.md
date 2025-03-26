# FirstRespondersChatbot

A chatbot system for first responders using Retrieval-Augmented Generation (RAG) to provide accurate information about emergency procedures and protocols.

## Project Overview

The FirstRespondersChatbot is designed to assist first responders by providing quick and accurate information about emergency procedures, protocols, and best practices. The system uses a combination of:

1. **Fine-tuned Language Model**: Models available include:
   - **TinyLlama 1.1B**: Our primary model optimized for fast inference ("tinyllama-1.1b-first-responder-fast")
   - **Llama 3.1 1B**: Option for state-of-the-art performance
   - Flan-T5 model: Legacy support for older deployments
2. **Retrieval-Augmented Generation (RAG)**: Enhances responses by retrieving relevant information from a document store
3. **Hybrid Retrieval**: Combines semantic search with keyword-based retrieval for better results

## Current Model Configuration

The system is now configured to use the "tinyllama-1.1b-first-responder-fast" model exclusively. This model provides an excellent balance of response quality and speed, particularly on devices with limited resources.

## Recent Optimizations (v1.1.0)

Several important optimizations have been implemented to improve system performance and reliability:

1. **Enhanced Memory Management**: Better handling of CUDA memory, with graceful degradation on OOM errors
2. **Improved Error Handling**: More robust error recovery and informative error messages
3. **Enhanced File Processing**: Better support for various file formats and encoding detection
4. **Optimized Prompting**: Refined prompt templates for TinyLlama
5. **Performance Tracking**: Basic metrics for request processing and system uptime
6. **Apple Silicon Support**: Improved performance on M-series chips

## Features

- **Document Processing**: Upload and process PDF, DOCX, TXT, HTML and MD files containing first responder information
- **Natural Language Queries**: Ask questions in natural language about emergency procedures
- **Context-Aware Responses**: Get responses that include citations to the source documents
- **Hardware Optimization**: Support for Apple Silicon (M1/M2/M3/M4), NVIDIA GPUs, and CPU-only environments
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
   # Download from the releases page or build your own with train.py
   ```

## Usage

### Command-Line Interface

To start an interactive chat session:

```bash
python -m firstresponders_chatbot cli chat
```

To ask a single question:

```bash
python -m firstresponders_chatbot cli query "What is the protocol for CPR?"
```

### Document Processing

To preprocess documents:

```bash
python -m firstresponders_chatbot preprocess --docs-dir ./docs --output-dir ./data
```

### Dataset Creation

To create a training dataset:

```bash
python create_dataset.py --model_format tinyllama
```

### Model Training

The system supports various training configurations optimized for different hardware:

#### TinyLlama Training (Recommended)

```bash
python train.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output_dir tinyllama-1.1b-first-responder-fast --max_seq_length 512 --gradient_accumulation_steps 8
```

#### Training on Apple Silicon

```bash
python train.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output_dir tinyllama-1.1b-first-responder-fast --max_seq_length 512 --lora_r 8 --lora_alpha 16 --gradient_accumulation_steps 8 --fp16
```

#### Advanced Options

- `--rebuild_dataset`: Rebuild the dataset with improved processing techniques
- `--skip_preprocessing`: Skip tokenization preprocessing (useful for troubleshooting)
- `--fp16`: Use mixed precision training (recommended for NVIDIA GPUs)
- `--load_in_4bit`: Load model in 4-bit precision for memory efficiency (recommended)
- `--max_train_samples`: Limit training samples for faster iteration during development

### Server

To start the REST API server:

```bash
python -m firstresponders_chatbot server --host 0.0.0.0 --port 8000
```

This will start the server using the default TinyLlama model directory.

## API Documentation

The REST API provides the following endpoints:

- `GET /api/health`: Health check endpoint with server metrics
- `POST /api/upload`: Upload a document file (PDF, DOCX, TXT, HTML, MD)
- `POST /api/query`: Query the RAG system
- `POST /api/clear`: Clear the document index
- `GET /api/files`: Get a list of indexed files
- `GET /api/files/<filename>`: Access uploaded files directly
- `POST /api/remove-file`: Remove a specific file
- `POST /api/clear-session`: Clear a specific session

## Training with Llama 3.1 1B (Alternative)

To train with Llama 3.1 1B as an alternative to TinyLlama:

### Creating a Dataset with Llama Format

First, create a dataset specifically formatted for Llama:

```bash
python create_dataset.py --model_format llama
```

### Training the Model

To train the model with the Llama configuration:

```bash
python train_llama.py --fp16
```

For more details, see the TRAINING_GUIDE.md file.

## Deployment

The FirstRespondersChatbot can be deployed in various environments:

1. **Local Deployment**: Run the CLI tool or server on a local machine
2. **Cloud Deployment**: Deploy the server on a cloud VM
3. **Containerized Deployment**: Docker support included (see Dockerfile)

## Contributing

We welcome contributions to the FirstRespondersChatbot! Please see CONTRIBUTING.md for guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TinyLlama team for providing an efficient base model
- Hugging Face for their Transformers library
- Haystack team for their RAG components