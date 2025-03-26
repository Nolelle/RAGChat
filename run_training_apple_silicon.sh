#!/bin/zsh

# Simple end-to-end training script for FirstRespondersChatbot on Apple Silicon
echo "======================================================================"
echo "🚒 FirstResponders Chatbot Training Pipeline for Llama 2 on Apple Silicon"
echo "======================================================================"

# Create dirs if they don't exist
mkdir -p data
mkdir -p trained-models/llama2-first-responder
mkdir -p docs

# Check if docs directory is empty
if [ -z "$(ls -A docs)" ]; then
    echo "⚠️  Warning: 'docs' directory appears to be empty. Please add your training documents there."
    echo "   Would you like to continue anyway? (y/n)"
    read response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Exiting. Please add documents to the 'docs' directory and try again."
        exit 1
    fi
fi

# Set environment variables for PyTorch MPS optimization
export PYTORCH_ENABLE_MPS_EAGER_FALLBACK=1
export OMP_NUM_THREADS=8
echo "✅ Set environment variables for optimal Apple Silicon performance"

# Set model parameters for Llama 2 on Apple Silicon
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
OUTPUT_DIR="trained-models/llama2-first-responder"
MAX_SEQ_LENGTH=2048
GRAD_ACCUM_STEPS=32
LEARNING_RATE=1e-4
BATCH_SIZE=1
EPOCHS=2
LORA_DROPOUT=0.1

echo "📋 Training Configuration:"
echo "   • Model: $MODEL_NAME"
echo "   • Output Directory: $OUTPUT_DIR"
echo "   • Max Sequence Length: $MAX_SEQ_LENGTH"
echo "   • Gradient Accumulation Steps: $GRAD_ACCUM_STEPS"
echo "   • Learning Rate: $LEARNING_RATE"
echo "   • Epochs: $EPOCHS"

# Start the training process with timing
echo "\n📊 Starting training pipeline..."
TOTAL_START_TIME=$(date +%s)

# Step 1: Run preprocessing
echo "\n🔍 Step 1/3: Running document preprocessing... (estimated time: 3-10 min)"
STEP1_START_TIME=$(date +%s)
python3 preprocess.py
STEP1_END_TIME=$(date +%s)
STEP1_DURATION=$((STEP1_END_TIME - STEP1_START_TIME))
echo "✅ Preprocessing completed in $((STEP1_DURATION / 60)) min $((STEP1_DURATION % 60)) sec"

# Step 2: Create the dataset for Llama 2
echo "\n🧩 Step 2/3: Creating dataset with Llama 2 format... (estimated time: 5-15 min)"
STEP2_START_TIME=$(date +%s)
python3 create_dataset.py
STEP2_END_TIME=$(date +%s)
STEP2_DURATION=$((STEP2_END_TIME - STEP2_START_TIME))
echo "✅ Dataset creation completed in $((STEP2_DURATION / 60)) min $((STEP2_DURATION % 60)) sec"

# Step 3: Run the training with the preprocessed dataset
echo "\n🧠 Step 3/3: Training Llama 2 model... (estimated time: 30-120 min)"
STEP3_START_TIME=$(date +%s)
python3 train.py \
    --model_name $MODEL_NAME \
    --max_train_samples 500 \
    --max_seq_length $MAX_SEQ_LENGTH \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $EPOCHS \
    --fp16 \
    --batch_size $BATCH_SIZE \
    --lora_dropout $LORA_DROPOUT \
    --mps_enable_eager_mode \
    --output_dir $OUTPUT_DIR
STEP3_END_TIME=$(date +%s)
STEP3_DURATION=$((STEP3_END_TIME - STEP3_START_TIME))
echo "✅ Training completed in $((STEP3_DURATION / 60)) min $((STEP3_DURATION % 60)) sec"

# Calculate total time
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo "\n======================================================================"
echo "🎉 Training pipeline completed in ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "📁 Trained model saved to: $OUTPUT_DIR"
echo "======================================================================"
echo "To use your model, update the model path in your application config." 