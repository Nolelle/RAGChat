# FirstRespondersChatbot

A RAG-based chatbot for firefighters and first responders, optimized for Apple Silicon.

## Table of Contents

- [Key Features](#key-features)
- [System Overview](#system-overview)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Training Guide](#training-guide)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Apple Silicon Optimization](#apple-silicon-optimization)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Key Features

1. **Phi-4-mini-instruct Model**:
   - Lightweight, high-performance open model from Microsoft
   - Strong reasoning and instruction-following capabilities
   - Optimized specifically for first responder domain knowledge

2. **RAG (Retrieval Augmented Generation)**: Uses your organization's protocol documents
3. **Optimized for Apple Silicon**: Engineered for performance on M-series chips
4. **Hybrid Retrieval**: Combines sparse and dense retrieval for better results
5. **Web and CLI Interfaces**: Multiple ways to interact with the system

## System Overview

The system leverages Retrieval Augmented Generation to provide accurate, contextual responses to first responder queries by retrieving information from your organization's protocols and manuals.

### Architecture

The FirstRespondersChatbot consists of three main subsystems:

1. **Document Processing**:
   - PDF and text document ingestion
   - Document cleaning and chunking
   - Metadata extraction and embedding

2. **Training Pipeline**:
   - Preprocessing: Optimized for multiple LLM architectures
   - Dataset creation: Generates training data in appropriate format
   - Training: Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning

3. **Inference**:
   - Hybrid RAG retrieval (sparse + dense)
   - Context optimization
   - Response generation
   - User-friendly interfaces (CLI or web)

```
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│  Document Pipeline  │      │  Training Pipeline  │      │  Inference System   │
│                     │      │                     │      │                     │
│ - PDF/Text Ingestion│      │ - Data Preprocessing│      │ - Query Processing  │
│ - Text Extraction   │──────│ - Dataset Creation  │      │ - Document Retrieval│
│ - Chunking          │      │ - Model Fine-tuning │      │ - Context Formation │
│ - Embedding         │      │ - Evaluation        │──────│ - Response Generation│
│ - Document Store    │      │                     │      │ - User Interface    │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘
```

## Setup and Installation

### Prerequisites

- Python 3.12+
- Hugging Face account (for downloading models)
- PyTorch with MPS support (for Apple Silicon)
- Node.js 18+ (for frontend)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/firstresponders-chatbot.git
   cd firstresponders-chatbot
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -e .
   ```

4. Set up Hugging Face token (recommended for rate limits, optional for Phi-4-mini):

   ```bash
   huggingface-cli login
   ```

5. Set up the frontend:

   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

## Usage

### Training

1. Place your training documents in the `training-docs/` directory.

2. Run the training script:

   ```bash
   ./run_training_apple_silicon.sh
   ```

   This script will:
   - Preprocess documents and create chunks
   - Generate a training dataset (compatible format)
   - Fine-tune the Phi-4-mini-instruct model with LoRA
   - Save the trained model to `trained-models/phi4-mini-first-responder`

### Running the Chatbot

Start the server:

```bash
# Run the server using the module execution pattern
python -m src.firstresponders_chatbot.rag.run_server
```

Access the web interface:

- Open your browser to <http://localhost:8000>

Use the Command-Line Interface (CLI):

```bash
# Start an interactive chat session
python -m src.firstresponders_chatbot.cli.cli chat

# Ask a single question
python -m src.firstresponders_chatbot.cli.cli query "Your question here?"
```

## Training Guide

### Dataset Preparation

First, preprocess your first responder documents:

```bash
# Use module execution for preprocessing
python -m src.firstresponders_chatbot.preprocessing.preprocessor --docs-dir ./training-docs --output-dir ./data
```

Then create a training dataset:

```bash
# Use module execution for dataset creation
python -m src.firstresponders_chatbot.training.dataset_creator --input-file ./data/preprocessed_data.json --output-file ./data/pseudo_data.json
```

### Training Command

```bash
# Use module execution for training
# Example: Train with Phi-4-mini-instruct
python -m src.firstresponders_chatbot.training.trainer --model_name microsoft/Phi-4-mini-instruct --output_dir trained-models/phi4-mini-first-responder
```

### Key Training Parameters

- `--model_name`: Base model to fine-tune (microsoft/Phi-4-mini-instruct)
- `--output_dir`: Directory to save the trained model
- `--batch_size`: Batch size for training (default: 1)
- `--gradient_accumulation_steps`: Steps to accumulate gradients (default: 32 for MPS, 16 for others)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--num_train_epochs`: Number of training epochs (default: 2)
- `--max_seq_length`: Maximum sequence length (default: 2048 for MPS, 3072 for others)
- `--train_test_split`: Fraction of data for evaluation (default: 0.1)

### Hardware-Specific Training Optimizations

#### Apple Silicon (M1/M2/M3/M4)

```bash
# For Phi-4-mini (uses our optimized script)
./run_training_apple_silicon.sh

# Manual configuration for Phi-4-mini (using module execution)
python -m src.firstresponders_chatbot.training.trainer --model_name microsoft/Phi-4-mini-instruct --output_dir trained-models/phi4-mini-first-responder --max_seq_length 2048 --gradient_accumulation_steps 32
```

#### NVIDIA GPUs

```bash
# For Phi-4-mini (using module execution)
python -m src.firstresponders_chatbot.training.trainer --model_name microsoft/Phi-4-mini-instruct --output_dir trained-models/phi4-mini-first-responder --fp16
```

## API Reference

The system exposes REST API endpoints for integration with other applications.

### Base URL

All endpoints are relative to the base URL: `http://localhost:8000`

### Endpoints

#### Health Check

**Endpoint:** `GET /api/health`

**Response:**

```json
{
  "status": "ok",
  "uptime": "3600.45 seconds",
  "requests_served": 42,
  "version": "1.1.0"
}
```

#### Upload File

**Endpoint:** `POST /api/upload`

**Request:**

- Content-Type: `multipart/form-data`
- Body:
  - `file`: The file to upload (PDF, TXT, MD, DOCX, HTML)
  - `session_id` (optional): Session identifier (default: "default")

**Response (Success):**

```json
{
  "status": "success",
  "message": "File 'example.pdf' uploaded and indexed successfully",
  "file_name": "example.pdf",
  "file_path": "uploads/12345_example.pdf",
  "session_id": "user123"
}
```

#### Query

**Endpoint:** `POST /api/query`

**Request:**

- Content-Type: `application/json`
- Body:

```json
{
  "query": "What should I do in case of a heart attack?",
  "session_id": "user123"
}
```

**Response (Success):**

```json
{
  "status": "success",
  "answer": "The generated answer from the model...",
  "context": [
    {
      "source": "uploads/12345_first_aid_manual.pdf",
      "content": "In case of a heart attack, call emergency services immediately...",
      "file_name": "first_aid_manual.pdf"
    }
  ],
  "query": "What should I do in case of a heart attack?",
  "session_id": "user123"
}
```

#### Clear Index

**Endpoint:** `POST /api/clear`

**Response (Success):**

```json
{
  "status": "success",
  "message": "Document index cleared successfully"
}
```

#### Get Indexed Files

**Endpoint:** `GET /api/files`

**Parameters:**

- `session_id` (optional): Session identifier (default: "default")

**Response (Success):**

```json
{
  "status": "success",
  "files": [
    {
      "name": "example.pdf",
      "path": "uploads/12345_example.pdf",
      "size": 1024,
      "type": "pdf",
      "last_modified": 1621345678.123
    }
  ],
  "session_id": "user123"
}
```

### JavaScript/React Example

```javascript
// Example: Upload a file with session ID
const uploadFile = async (file, sessionId = "default") => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('session_id', sessionId);
  
  try {
    const response = await fetch('http://localhost:8000/api/upload', {
      method: 'POST',
      body: formData,
    });
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error uploading file:', error);
    return { status: 'error', message: error.message };
  }
};

// Example: Send a query with session ID
const sendQuery = async (query, sessionId = "default") => {
  try {
    const response = await fetch('http://localhost:8000/api/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        query,
        session_id: sessionId 
      }),
    });
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error sending query:', error);
    return { status: 'error', message: error.message };
  }
};
```

## Project Structure

```
firstresponders-chatbot/
├── frontend/             # React frontend code
│   ├── src/              # Frontend source code
│   │   ├── components/   # React components
│   │   │   ├── ChatWindow.jsx
│   │   │   ├── MessageInput.jsx
│   │   │   └── Sidebar.jsx
│   │   ├── App.jsx       # Main application component
│   │   └── index.js      # Entry point
├── src/                  # Main Python package
│   └── firstresponders_chatbot/
│       ├── cli/          # Command Line Interface
│       │   └── cli.py    # CLI implementation
│       ├── preprocessing/# Document preprocessing
│       ├── rag/          # RAG system (retrieval, server, etc.)
│       │   ├── rag_system.py
│       │   ├── run_server.py # Server entry point
│       │   └── server.py     # Server implementation
│       ├── training/     # Model training pipeline
│       └── utils/        # Utility functions
├── trained-models/       # Saved fine-tuned models
├── training-docs/        # Source documents for training
├── data/                 # Processed data and datasets
├── uploads/              # Uploaded documents for RAG
├── project-documentation/# Detailed documentation
│   ├── API_DOCUMENTATION.md
│   └── TRAINING_GUIDE.md
├── run_training_apple_silicon.sh # Training script
├── setup_token.py        # Hugging Face token setup utility
├── train.py              # Training script entry point
├── pyproject.toml        # Project configuration and dependencies
└── README.md             # This file
```

### Key Components

- `src/firstresponders_chatbot/rag/run_server.py`: Entry point for the RAG server.
- `src/firstresponders_chatbot/cli/cli.py`: Entry point for the CLI.
- `src/firstresponders_chatbot/training/`: Training module.
- `src/firstresponders_chatbot/preprocessing/`: Document preprocessing module.
- `frontend/`: React-based user interface.

## Apple Silicon Optimization

The system is specifically optimized for Apple Silicon (M1/M2/M3/M4):

- **MPS Acceleration**: Leverages Metal Performance Shaders for faster inference
- **Memory Optimization**: Efficient memory management for M-series chips
- **Training Scripts**: Specialized scripts for efficient training on Apple Silicon
- **Model Selection**: Support for smaller models like Phi-4-mini that work well on Apple hardware

## Deployment

### Local Deployment

```bash
# Start the server
python -m src.firstresponders_chatbot.rag.run_server

# Access the web interface
# Open http://localhost:8000 in your browser
```

### Production Deployment

For production deployment, consider:

- Using Gunicorn or uWSGI as WSGI server
- Setting up NGINX as a reverse proxy
- Deploying with Docker for containerization
- Using managed services like AWS or GCP

Example Gunicorn deployment:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 "src.firstresponders_chatbot.rag.run_server:create_app()"
```

## Troubleshooting

### Common Issues and Solutions

#### "ValueError: You should supply an encoding or a list of encodings..."

- The dataset format doesn't match what the model expects
- Solution: Use `--skip_preprocessing` flag to let the trainer handle the conversion

#### "CUDA out of memory" or "MPS out of memory"

- Not enough GPU/MPS memory
- Solutions:
  - Decrease batch size
  - Increase gradient accumulation steps
  - Use a smaller model
  - Reduce sequence length with `--max_seq_length 512`

#### Slow training on Apple Silicon

- Solutions:
  - Reduce sequence lengths
  - Reduce gradient accumulation steps
  - Use smaller batch sizes

#### "Token not found" error

- Hugging Face token is not set up correctly
- Solution: Run `python setup_token.py` and follow the prompts

#### Server fails to start

- Check that all dependencies are installed
- Verify model path is correct
- Ensure port 8000 is not already in use
- Check logs for specific errors

## License

MIT

## Acknowledgements

- Microsoft for providing the Phi-4 models
- Hugging Face for model hosting and transformers library
- Haystack for document processing capabilities
