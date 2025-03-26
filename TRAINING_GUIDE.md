# FirstRespondersChatbot Training Guide

This guide provides detailed instructions on how to train, evaluate, and optimize the FirstRespondersChatbot model for different hardware configurations and use cases.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Dataset Preparation](#dataset-preparation)
- [Training Configuration](#training-configuration)
- [Hardware-Specific Optimizations](#hardware-specific-optimizations)
- [Two-Stage Training Process](#two-stage-training-process)
- [Evaluation Metrics](#evaluation-metrics)
- [Troubleshooting](#troubleshooting)
- [Advanced Configurations](#advanced-configurations)
- [TinyLlama Training](#tinyllama-training)

## Prerequisites

Before training the model, ensure you have:

- Python 3.9+ installed
- Required packages installed (run `pip install -e .` from the project root)
- NLTK resources downloaded (will be downloaded automatically on first run)
- Sufficient disk space for model checkpoints (~1-2GB for TinyLlama)
- Appropriate hardware (see [Hardware-Specific Optimizations](#hardware-specific-optimizations))

## Dataset Preparation

### Document Preprocessing

First, preprocess your first responder documents:

```bash
python preprocess.py --docs-dir ./docs --output-dir ./data
```

This will:
- Extract text from PDF, TXT, and MD files
- Clean and normalize the text
- Split documents into manageable chunks
- Store preprocessed data in `./data/preprocessed_data.json`

### Dataset Creation

Next, create a training dataset:

```bash
python create_dataset.py --input-file ./data/preprocessed_data.json --output-file ./data/pseudo_data.json
```

This process:
- Generates question-answer pairs from the preprocessed documents
- Applies quality filtering to ensure good training examples
- Creates a balanced dataset for training
- Outputs the dataset to `./data/pseudo_data.json`

### Model-Specific Dataset Formatting

For different model architectures, you can specify the appropriate format:

```bash
# For Llama models
python create_dataset.py --model_format llama

# For TinyLlama models
python create_dataset.py --model_format tinyllama
```

### Rebuilding the Dataset with Improved Techniques

To rebuild the dataset with enhanced processing:

```bash
python train.py --rebuild_dataset --output_dir tinyllama-1.1b-first-responder-fast
```

This will apply:
- Better text cleaning and deduplication
- Semantic similarity grouping for context
- More advanced question generation
- Improved answer extraction

## Training Configuration

The basic training command is:

```bash
python train.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output_dir tinyllama-1.1b-first-responder-fast
```

### Key Parameters

- `--model_name`: The base model to fine-tune (options include: TinyLlama/TinyLlama-1.1B-Chat-v1.0, meta-llama/Llama-3.1-1B-Instruct)
- `--output_dir`: Directory to save the trained model
- `--batch_size`: Batch size for training (default: 1, increase for more GPU memory)
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients (default: 32)
- `--learning_rate`: Learning rate (default: 3e-5)
- `--num_train_epochs`: Number of training epochs (default: 8)
- `--max_seq_length`: Maximum sequence length (adjust based on model)
- `--train_test_split`: Fraction of data to use for evaluation (default: 0.1)

## Hardware-Specific Optimizations

### Apple Silicon (M1/M2/M3/M4)

For Apple Silicon, use these settings for optimal performance:

```bash
python train.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output_dir tinyllama-1.1b-first-responder-fast --max_seq_length 512 --gradient_accumulation_steps 8
```

- The system automatically detects Apple Silicon and enables MPS acceleration
- Avoid `--fp16` as it's not fully compatible with MPS
- Reduce sequence lengths to improve memory usage
- Consider using TinyLlama or Llama 3.1 1B for faster training

### NVIDIA GPUs

For NVIDIA GPUs, leverage these options:

```bash
python train.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output_dir tinyllama-1.1b-first-responder-fast --fp16 --load_in_4bit
```

- The `--fp16` flag enables mixed precision training for faster computation
- The `--load_in_4bit` flag enables 4-bit quantization for memory efficiency (requires bitsandbytes)
- Adjust batch size and gradient accumulation based on available VRAM

### CPU-Only Environment

For CPU-only training:

```bash
python train.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output_dir tinyllama-1.1b-first-responder-fast --num_train_epochs 3
```

- Use TinyLlama for CPU training
- Reduce sequence lengths and epochs for faster completion

## TinyLlama Training

TinyLlama is an efficient 1.1B parameter model that provides fast inference while maintaining good quality responses. The model currently in production is "tinyllama-1.1b-first-responder-fast", which was optimized for fast inference on resource-constrained devices.

To train a TinyLlama model:

```bash
python train.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output_dir tinyllama-1.1b-first-responder-fast --max_seq_length 512 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --num_train_epochs 3 --learning_rate 3e-4 --gradient_accumulation_steps 8
```

This configuration:
- Uses the smaller sequence length to reduce memory usage
- Applies LoRA (Low-Rank Adaptation) for more efficient fine-tuning
- Uses a slightly higher learning rate to accelerate training
- Reduces the number of epochs to minimize training time

## Evaluation Metrics

The training process automatically evaluates the model using:

- **ROUGE Scores**: Measures overlap between generated and reference text
- **BLEU Score**: Evaluates translation quality
- **Generated Length**: Tracks the average length of generated responses

To view evaluation metrics:
1. Check the training logs in the terminal output
2. Examine the `trainer_state.json` file in the output directory
3. Use TensorBoard for visualization:
   ```bash
   tensorboard --logdir ./your-model-dir/runs
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

## Advanced Configurations

### Custom Tokenization

For specialized vocabularies:

```bash
python train.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output_dir tinyllama-1.1b-first-responder-fast --tokenizer_name your-custom-tokenizer
```

### Fine-tuning from Your Own Checkpoint

To continue training from a checkpoint:

```bash
python train.py --model_name ./path/to/your/checkpoint --output_dir tinyllama-1.1b-first-responder-continued
```

### Dataset Filtering

To train on a subset of data:

```bash
python create_dataset.py --input-file ./data/preprocessed_data.json --output-file ./data/filtered_data.json --max-examples-per-doc 2
```

### Using the Trained Model in RAG Pipeline

After training, integrate your model with the Haystack RAG pipeline by configuring the model directory in the RAGSystem initialization:

```python
# In your server.py or other integration point
from src.firstresponders_chatbot.rag.rag_system import RAGSystem

# Initialize with your chosen model
rag_system = RAGSystem(model_dir="tinyllama-1.1b-first-responder-fast")
```

## Conclusion

This training guide provides comprehensive instructions for training the FirstRespondersChatbot model with optimal configurations for different hardware and use cases. By following these guidelines, you can create a high-quality model that provides accurate and helpful responses to first responder queries.