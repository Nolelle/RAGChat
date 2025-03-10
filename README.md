# First Responder RAG Chatbot

A specialized chatbot system that helps firefighters, EMTs, and other first responders quickly access critical information from procedure manuals, protocols, and training materials using Retrieval-Augmented Generation (RAG).

## Features

The system provides real-time, context-aware responses by combining document retrieval with AI-powered text generation:

- **Intelligent Query Response**: Delivers accurate answers using official manuals, SOPs, and documentation
- **Dynamic Knowledge Base**: Supports PDF and text file uploads for expanding the system's knowledge
- **Real-time Information Access**: Uses RAG to retrieve relevant information chunks from stored documents
- **User-friendly Interface**: Offers both CLI and web-based interfaces for easy interaction
- **Scalable Architecture**: Designed to grow with additional domains and evolving guidelines

## Technical Architecture

### Core Components

The system is built using modern AI and information retrieval technologies:

- **Document Processing**: Converts and chunks documents while preserving semantic context
- **Embedding System**: Generates high-quality vector representations using instructor-xl (768 dimensions)
- **Vector Store**: Uses FAISS for efficient similarity search and retrieval
- **Generation Model**: Leverages flan-t5-base for coherent response generation
- **Web Interface**: Built with Gradio for intuitive interaction

### Project Structure

```
chatbot-rag-firefight/
├── pyproject.toml          # Project metadata and dependencies
├── src/                    # Source code directory
│   ├── data_processing/   # Document handling and preprocessing
│   ├── model/            # RAG model implementation
│   ├── retrieval/        # Vector storage and search
│   ├── cli/             # Command-line interface
│   └── web/             # Web interface (Gradio app)
├── data/                 # Data directory (git-ignored)
│   ├── raw/             # Original documents
│   ├── processed/       # Processed text chunks
│   └── embeddings/      # Stored document embeddings
├── tests/               # Test suite
└── README.md            # Project documentation
```

## System Requirements and Performance

### Hardware Requirements
The system is optimized for local development on macOS with:
- Apple M1/M2 series processor or equivalent
- Minimum 16GB RAM recommended
- At least 256GB available SSD storage
- macOS Monterey or newer

### Performance Targets
The system is designed for efficient local processing:

- Query Response Time: < 3 seconds total
  - Embedding generation: < 200ms
  - Vector search: < 100ms
  - Response generation: < 2.5 seconds
- Document Processing: < 5 seconds per page
- Peak Memory Usage: < 4GB
- Storage Usage: < 1GB per 500 pages of documents

For detailed technical specifications, including optimization strategies, monitoring setup, and development configurations, see our [Technical Documentation](docs/TECHNICAL.md).

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chatbot-rag-firefight.git
   cd chatbot-rag-firefight
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e ".[dev]"  # Includes development dependencies
   ```

4. Create required data directories:
   ```bash
   mkdir -p data/{raw,processed,embeddings}
   ```

5. Run initial tests:
   ```bash
   pytest
   ```

## Using the CLI

The system includes a command-line interface powered by Haystack for document processing and querying:

### Processing Documents

To process a single document:
```bash
python main.py process-document path/to/your/document.pdf
```

To process all documents in a directory:
```bash
python main.py process-directory path/to/your/documents/
```

### Searching for Information

To query the knowledge base:
```bash
python main.py query "What is the procedure for handling a hazardous materials incident?"
```

### Interactive Chat Mode

For continuous interaction with the knowledge base:
```bash
python main.py chat
```

## Development

The project uses modern Python tools and practices:

- **Python Version**: 3.12+
- **Code Style**: Black formatter and Ruff linter
- **Testing**: Pytest with coverage reporting
- **Documentation**: Sphinx with ReadTheDocs theme
- **Type Hints**: Comprehensive typing throughout

### Installing Optional Dependencies

For web interface development:
```bash
pip install -e ".[web]"
```

For documentation building:
```bash
pip install -e ".[docs]"
```

## Testing

The project includes comprehensive test coverage:

- **Unit Tests**: Component-level accuracy validation
- **Integration Tests**: End-to-end pipeline verification
- **Performance Tests**: Response time and resource usage benchmarks
- **Validation Tests**: Real-world usage scenarios

Run the full test suite:
```bash
pytest --cov=src tests/
```

## License

[Your License Here]
