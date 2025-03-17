# FirstRespondersChatbot

A chatbot system for first responders using Retrieval-Augmented Generation (RAG) to provide accurate information about emergency procedures and protocols.

## Project Overview

The FirstRespondersChatbot is designed to assist first responders by providing quick and accurate information about emergency procedures, protocols, and best practices. The system uses a combination of:

1. **Fine-tuned Language Model**: A Flan-T5 model fine-tuned on first responder documentation
2. **Retrieval-Augmented Generation (RAG)**: Enhances responses by retrieving relevant information from a document store

## Features

- **Document Processing**: Upload and process PDF, TXT, and MD files containing first responder information
- **Natural Language Queries**: Ask questions in natural language about emergency procedures
- **Context-Aware Responses**: Get responses that include citations to the source documents
- **Multiple Interfaces**:
  - Command-line interface (CLI) for direct interaction
  - REST API server for integration with web applications
  - Web UI (coming soon)

## Installation

### Prerequisites

- Python 3.9+
- pip

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

To train the model:

```bash
python train.py
```

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
│       ├── rag/              # RAG system
│       └── training/         # Model training
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the Transformers library
- Haystack for the RAG components
- All contributors to the project
