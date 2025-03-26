# FirstRespondersChatbot

A RAG-based chatbot for firefighters and first responders, optimized for Apple Silicon.

## Key Features

1. **Fine-tuned Language Model**: Meta's Llama 2 model
2. **RAG (Retrieval Augmented Generation)**: Uses your organization's protocol documents
3. **Optimized for Apple Silicon**: Engineered for performance on M-series chips
4. **Optimized Prompting**: Refined prompt templates for Llama 2
5. **Interleaved Responses**: Mixes domain-specific advice with safety information

## System Overview

The system is configured to use Meta's Llama 2 model, which provides an excellent balance of response quality and speed, particularly on devices with limited resources.

### Architecture

1. **Document Processing**:
   - PDF and text document ingestion
   - Document cleaning and chunking
   - Metadata extraction

2. **Training Pipeline**:
   - Preprocessing: Optimized for Llama 2
   - Dataset creation: Generates training data in Llama 2 format
   - Training: Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning

3. **Inference**:
   - RAG-enhanced responses
   - User-friendly interface options (CLI or web)

## Setup and Installation

### Prerequisites

- Python 3.10+
- Hugging Face account with access to Llama 2
- PyTorch with MPS support (for Apple Silicon)

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

4. Set up Hugging Face token (for accessing Llama 2):

   ```bash
   python setup_token.py
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
   - Generate a training dataset with Llama 2 formatting
   - Fine-tune a Llama 2 model with LoRA
   - Save the trained model to `trained-models/llama2-first-responder`

### Running the Chatbot

Start the server:

```bash
python server.py
```

This will start the server using the Llama 2 model.

Access the web interface:
- Open your browser to http://localhost:8000

## Advanced Configuration

See `TRAINING_GUIDE.md` for detailed training options and configurations.

## Project Structure

- `preprocess.py`: Document preprocessing
- `create_dataset.py`: Training data creation
- `train.py`: Model training
- `run_training_apple_silicon.sh`: End-to-end training script
- `server.py`: Web interface

## License

MIT

## Acknowledgements

- Meta for providing the Llama 2 model
- Hugging Face for model hosting and transformers library
- Haystack for document processing capabilities
