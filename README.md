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

1. **Llama 3 8B Instruct Model**:
   - High-quality instruction-following capabilities
   - Excellent context understanding and response accuracy
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
- Hugging Face account with access to Llama models
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

4. Set up Hugging Face token (for accessing Llama models):

   ```bash
   python setup_token.py
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

1. Place your training documents in the `docs/` directory.

2. Run the training script:

   ```bash
   ./run_training_apple_silicon.sh
   ```

   This script will:
   - Preprocess documents and create chunks
   - Generate a training dataset with Llama 3 formatting
   - Fine-tune Llama 3 8B Instruct model with LoRA
   - Save the trained model to `trained-models/llama3-first-responder`

### Running the Chatbot

Start the server:

```bash
python server.py --model trained-models/llama3-first-responder
```

Access the web interface:
- Open your browser to http://localhost:8000

## Training Guide

### Dataset Preparation

First, preprocess your first responder documents:

```bash
python preprocess.py --docs-dir ./docs --output-dir ./data
```

Then create a training dataset formatted for Llama 3:

```bash
python create_dataset.py --input-file ./data/preprocessed_data.json --output-file ./data/pseudo_data.json
```

### Training Command

```bash
# Train with Llama 3 8B Instruct
python train.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --output_dir llama3-first-responder
```

### Key Training Parameters

- `--model_name`: Base model to fine-tune (meta-llama/Meta-Llama-3-8B-Instruct)
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
# For Llama 3 8B (uses our optimized script)
./run_training_apple_silicon.sh

# Manual configuration for Llama 3
python train.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --output_dir llama3-first-responder --max_seq_length 2048 --gradient_accumulation_steps 32
```

#### NVIDIA GPUs

```bash
# For Llama 3 8B
python train.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --output_dir llama3-first-responder --fp16 --load_in_4bit
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
  "status": "ok"
}
```

#### Upload File

**Endpoint:** `POST /api/upload`

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file`: The file to upload (PDF, TXT, or MD)

**Response (Success):**
```json
{
  "status": "success",
  "message": "File 'example.pdf' uploaded and indexed successfully",
  "file_path": "uploads/12345_example.pdf"
}
```

#### Query

**Endpoint:** `POST /api/query`

**Request:**
- Content-Type: `application/json`
- Body:
```json
{
  "query": "What should I do in case of a heart attack?"
}
```

**Response (Success):**
```json
{
  "status": "success",
  "answer": "The generated answer from the model...",
  "context": [
    {
      "file_name": "first_aid_manual.pdf",
      "snippet": "In case of a heart attack, call emergency services immediately..."
    }
  ],
  "query": "What should I do in case of a heart attack?"
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

**Response (Success):**
```json
{
  "status": "success",
  "files": [
    {
      "name": "example.pdf",
      "path": "uploads/12345_example.pdf",
      "size": 1024,
      "type": "pdf"
    }
  ]
}
```

### JavaScript/React Example

```javascript
// Example: Upload a file
const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
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

// Example: Send a query
const sendQuery = async (query) => {
  try {
    const response = await fetch('http://localhost:8000/api/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query }),
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

- `src/firstresponders_chatbot/`: Main package
  - `preprocessing/`: Document preprocessing
  - `training/`: Model training
  - `rag/`: RAG implementation
  - `cli/`: Command-line interface
  - `utils/`: Utility functions
- `frontend/`: React web interface
- `docs/`: Place for your first responder documents
- `data/`: Processed data and datasets
- `trained-models/`: Saved model checkpoints
- `server.py`: Main server script
- `preprocess.py`: Document preprocessing script
- `create_dataset.py`: Dataset creation script
- `train.py`: Model training script
- `run_training_apple_silicon.sh`: End-to-end training script

## Apple Silicon Optimization

The system is specifically optimized for Apple Silicon (M1/M2/M3/M4):

- **MPS Acceleration**: Leverages Metal Performance Shaders for faster inference
- **Quantization**: Supports 4-bit and 8-bit quantization for efficient memory usage
- **Memory Optimization**: Efficient memory management for M-series chips
- **Training Scripts**: Specialized scripts for efficient training on Apple Silicon
- **Model Selection**: Support for smaller models (TinyLlama, Llama 3.1 1B) that work well on Apple hardware

## Deployment

### Local Deployment

```bash
# Start the server
python server.py

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

#### "CUDA out of memory"
- Not enough GPU memory
- Solutions:
  - Decrease batch size
  - Increase gradient accumulation steps
  - Use a smaller model
  - Enable 4-bit quantization with `--load_in_4bit`

#### Slow training on Apple Silicon
- Solutions:
  - Reduce sequence lengths
  - Reduce gradient accumulation steps
  - Use smaller batch sizes
  - Optimize quantization settings

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

- Meta for providing the Llama models
- Hugging Face for model hosting and transformers library
- Haystack for document processing capabilities
