# FirstRespondersChatbot

A Retrieval-Augmented Generation (RAG) chatbot for first responders using Haystack 2.0 and fine-tuned Flan-T5-Small.

## Project Overview

This project implements a specialized chatbot system for first responders, designed to provide accurate and contextual information for emergency scenarios. The system is built in three phases:

1. **Phase 1**: Document preprocessing and model fine-tuning
2. **Phase 2**: Command-line interface for querying the fine-tuned model
3. **Phase 3**: Full RAG system with a React frontend for file uploads and contextual queries

## Setup and Installation

### Prerequisites

- Python 3.12 or later
- pip or another Python package manager
- Node.js and npm (for Phase 3 - React frontend)

### Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd FirstRespondersChatbot
   ```

2. Set up a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install .  # For minimal installation
   pip install .[web]  # For web-related dependencies (Phase 3)
   pip install .[dev]  # For development dependencies
   ```

## Phase 1: Document Preprocessing and Model Fine-Tuning

### Prepare Documents

Place your first responder document files (PDF, TXT, MD) in the `docs/` directory. These will be used for preprocessing and model fine-tuning.

### Run the Preprocessing Script

```bash
python preprocess.py
```

This script will:
- Convert documents into text
- Split them into meaningful chunks
- Save the processed data to `data/preprocessed_data.json`

### Create the Training Dataset

```bash
python create_dataset.py
```

This script will:
- Take the preprocessed chunks
- Generate question-answer pairs for fine-tuning
- Save the dataset to `data/pseudo_data.json`

### Fine-tune the Model

```bash
python train.py
```

This script will:
- Load the Flan-T5-Small model
- Fine-tune it on the generated dataset
- Automatically detect and use available hardware (NVIDIA GPU or Apple Silicon)
- Save the fine-tuned model to `flan-t5-first-responder/`

## Phase 2: CLI Interface

After fine-tuning, you can use the CLI to interact with the model:

```bash
python cli.py
```

This provides a command-line interface to ask questions and get responses from your fine-tuned model.

## Phase 3: RAG System with Web Interface

### Start the Server

```bash
python server.py
```

This will start a Flask server that handles:
- File uploads for the RAG system
- Queries from the frontend
- Responses generated with contextual information

### Use the React Frontend

Navigate to the React app directory and start the development server:

```bash
cd rag-frontend
npm install
npm start
```

The web interface allows users to:
- Upload PDF and text files
- Ask questions
- Receive answers generated using relevant context from the uploaded documents

## Project Structure

```
FirstRespondersChatbot/
├── docs/                  # Raw document files
├── uploads/               # Temporary storage for user uploads
├── flan-t5-first-responder/ # Fine-tuned model storage
├── data/                  # Processed data and datasets
├── preprocess.py          # Document preprocessing script
├── create_dataset.py      # Dataset creation script
├── train.py               # Model fine-tuning script
├── cli.py                 # Command-line interface
├── rag_backend.py         # RAG system backend
├── server.py              # Flask server
├── rag-frontend/          # React frontend
│   └── src/
│       ├── App.js
│       └── (other React files)
└── pyproject.toml         # Project dependencies and configuration
```

## License

[Your chosen license]

## Acknowledgements

This project utilizes several open-source libraries:
- Haystack 2.0 for information retrieval
- Hugging Face Transformers for the Flan-T5 model
- Flask and React for the web interface
