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

## Prerequisites

Before training the model, ensure you have:

- Python 3.9+ installed
- Required packages installed (run `pip install -e .` from the project root)
- NLTK resources downloaded (will be downloaded automatically on first run)
- Sufficient disk space for model checkpoints (~2-4GB for flan-t5-large)
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

### Rebuilding the Dataset with Improved Techniques

To rebuild the dataset with enhanced processing:

```bash
python train.py --rebuild_dataset --output_dir flan-t5-large-first-responder
```

This will apply:
- Better text cleaning and deduplication
- Semantic similarity grouping for context
- More advanced question generation
- Improved answer extraction

## Training Configuration

The basic training command is:

```bash
python train.py --model_name google/flan-t5-large --output_dir flan-t5-large-first-responder
```

### Key Parameters

- `--model_name`: The base model to fine-tune (options: flan-t5-small, flan-t5-base, flan-t5-large, flan-t5-xl, flan-t5-xxl)
- `--output_dir`: Directory to save the trained model
- `--batch_size`: Batch size for training (default: 1, increase for more GPU memory)
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients (default: 32)
- `--learning_rate`: Learning rate (default: 3e-5)
- `--num_train_epochs`: Number of training epochs (default: 8)
- `--max_source_length`: Maximum input length (default: 384, recommend: 256 for faster training)
- `--max_target_length`: Maximum output length (default: 96, recommend: 64 for faster training)
- `--train_test_split`: Fraction of data to use for evaluation (default: 0.1)

## Hardware-Specific Optimizations

### Apple Silicon (M1/M2/M3)

For Apple Silicon, use these settings for optimal performance:

```bash
python train.py --model_name google/flan-t5-large --output_dir flan-t5-large-first-responder --freeze_encoder --max_source_length 256 --max_target_length 64 --gradient_accumulation_steps 16
```

- The system automatically detects Apple Silicon and enables MPS acceleration
- Avoid `--fp16` as it's not fully compatible with MPS
- Reduce sequence lengths to improve memory usage
- Consider using a smaller model (flan-t5-base) for faster training

### NVIDIA GPUs

For NVIDIA GPUs, leverage these options:

```bash
python train.py --model_name google/flan-t5-large --output_dir flan-t5-large-first-responder --fp16 --load_in_8bit
```

- The `--fp16` flag enables mixed precision training for faster computation
- The `--load_in_8bit` flag enables 8-bit quantization for memory efficiency (requires bitsandbytes)
- Adjust batch size and gradient accumulation based on available VRAM

### CPU-Only Environment

For CPU-only training:

```bash
python train.py --model_name google/flan-t5-small --output_dir flan-t5-small-first-responder --freeze_encoder --num_train_epochs 3
```

- Use smaller models (flan-t5-small or flan-t5-base)
- Reduce sequence lengths and epochs for faster completion
- Consider freezing the encoder to speed up training

## Two-Stage Training Process

For optimal results, we recommend a two-stage training approach:

### Stage 1: Freeze Encoder

```bash
python train.py --model_name google/flan-t5-large --output_dir flan-t5-large-first-responder-stage1 --freeze_encoder --num_train_epochs 3
```

This stage:
- Freezes the encoder layers to focus training on the decoder
- Requires less memory and trains faster
- Quickly adapts the model to the first responder domain

### Stage 2: Full Model Fine-tuning

```bash
python train.py --model_name ./flan-t5-large-first-responder-stage1 --output_dir flan-t5-large-first-responder-final --num_train_epochs 5
```

This stage:
- Takes the model from Stage 1 and fine-tunes all layers
- Allows the encoder to adapt to the specific language of first responder documents
- Creates a more powerful and accurate model

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
   tensorboard --logdir flan-t5-large-first-responder/runs
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
  - Enable 8-bit quantization with `--load_in_8bit`

#### "None of the inputs have requires_grad=True"
- Warning when using gradient checkpointing with frozen encoder
- Solution: This is expected behavior when using `--freeze_encoder`

#### Slow training on Apple Silicon
- Solutions:
  - Use `--freeze_encoder` flag
  - Reduce sequence lengths
  - Reduce gradient accumulation steps

## Advanced Configurations

### Custom Tokenization

For specialized vocabularies:

```bash
python train.py --model_name google/flan-t5-large --output_dir flan-t5-large-first-responder --tokenizer_name your-custom-tokenizer
```

### Fine-tuning from Your Own Checkpoint

To continue training from a checkpoint:

```bash
python train.py --model_name ./path/to/your/checkpoint --output_dir flan-t5-large-first-responder-continued
```

### Dataset Filtering

To train on a subset of data:

```bash
python create_dataset.py --input-file ./data/preprocessed_data.json --output-file ./data/filtered_data.json --max-examples-per-doc 2
```

### Using the Trained Model in RAG Pipeline

After training, integrate your model with the Haystack RAG pipeline:

```python
from haystack import Pipeline
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.retrievers import InMemoryBM25Retriever

generator = HuggingFaceLocalGenerator(
    model="./flan-t5-large-first-responder-final",
    task="text2text-generation"
)

# Set up retriever and pipeline as usual
# ...
```

## Conclusion

This training guide provides comprehensive instructions for training the FirstRespondersChatbot model with optimal configurations for different hardware and use cases. By following these guidelines, you can create a high-quality model that provides accurate and helpful responses to first responder queries. 