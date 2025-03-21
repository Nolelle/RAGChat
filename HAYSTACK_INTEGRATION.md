# Haystack 2.0 Integration Guide

This guide explains how to integrate your trained FirstRespondersChatbot model with Haystack 2.0 for building effective Retrieval-Augmented Generation (RAG) pipelines.

## Introduction to Haystack 2.0

Haystack 2.0 provides a flexible framework for building production-ready RAG systems. The FirstRespondersChatbot is designed to work seamlessly with Haystack's components for document retrieval and generation.

## Model Selection

The FirstRespondersChatbot project supports multiple models:

- **TinyLlama 1.1B** (default, "tinyllama-1.1b-first-responder-fast"): Optimized for fast inference
- **Llama 3.1 1B**: Excellent quality with moderate speed
- **Phi-3 Mini**: High quality, slightly slower
- **Flan-T5**: Original model architecture, multiple sizes available

Choose the model that best fits your performance and quality requirements.

## Basic RAG Pipeline

Here's a basic example of setting up a RAG pipeline with your trained model:

```python
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document

# 1. Set up document store
document_store = InMemoryDocumentStore()

# 2. Add documents to the store
document_store.write_documents([
    Document(content="CPR should be performed at a rate of 100-120 compressions per minute."),
    Document(content="For adult CPR, compress to a depth of at least 2 inches (5 cm)."),
    # Add your first responder documents here
])

# 3. Set up the generator using your fine-tuned model
generator = HuggingFaceLocalGenerator(
    model="./tinyllama-1.1b-first-responder-fast",  # Path to your trained model
    task="text-generation",  # Use "text2text-generation" for Flan-T5 models
    generation_kwargs={
        "max_new_tokens": 100,
        "temperature": 0.7,
        "do_sample": True,
    }
)

# 4. Create a prompt template
template = """
Answer the following question based on the context provided. Be concise and accurate.

Context: 
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ query }}

Answer:
"""

# 5. Build the RAG pipeline
pipe = Pipeline()
pipe.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component("llm", generator)
pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

# 6. Run the pipeline
response = pipe.run({
    "prompt_builder": {"query": "What is the correct rate for CPR?"},
    "retriever": {"query": "What is the correct rate for CPR?"}
})

print(response["llm"]["replies"][0])
```

## Hybrid Retrieval Pipeline

For better retrieval performance, set up a hybrid retrieval pipeline:

```python
from haystack import Pipeline
from haystack.components.retrievers import BM25Retriever, EmbeddingRetriever
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.joiners import DocumentJoiner
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

# 1. Set up document store with embeddings
document_store = InMemoryDocumentStore()

# 2. Create embedding model for documents
embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Process and index your documents with embeddings
docs = [Document(content="...") for doc in your_documents]
docs_with_embeddings = embedder.run(docs)["documents"]
document_store.write_documents(docs_with_embeddings)

# 4. Create hybrid retrieval pipeline
pipe = Pipeline()

# Add both sparse and dense retrievers
pipe.add_component("bm25_retriever", BM25Retriever(document_store=document_store, top_k=5))
pipe.add_component("embedding_retriever", EmbeddingRetriever(document_store=document_store, top_k=5))

# Join the results
pipe.add_component("joiner", DocumentJoiner(join_mode="concatenate", remove_duplicates=True))
pipe.connect("bm25_retriever", "joiner")
pipe.connect("embedding_retriever", "joiner")

# Re-rank the combined results
pipe.add_component("ranker", TransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=3))
pipe.connect("joiner", "ranker")

# Add the prompt builder and generator
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component("llm", generator)
pipe.connect("ranker", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

# Run the pipeline
response = pipe.run({
    "bm25_retriever": {"query": "What is the correct rate for CPR?"},
    "embedding_retriever": {"query": "What is the correct rate for CPR?"}
})
```

## Model-Specific Configurations

### TinyLlama Configuration

For TinyLlama models:

```python
generator = HuggingFaceLocalGenerator(
    model="./tinyllama-1.1b-first-responder-fast",
    task="text-generation",
    generation_kwargs={
        "max_new_tokens": 250,
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.2,
        "do_sample": True,
    }
)
```

### Llama 3.1 Configuration

For Llama 3.1 models:

```python
generator = HuggingFaceLocalGenerator(
    model="./llama-3.1-1b-first-responder",
    task="text-generation",
    generation_kwargs={
        "max_new_tokens": 350,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.2,
        "do_sample": True,
    }
)
```

### Phi-3 Configuration

For Phi-3 models:

```python
generator = HuggingFaceLocalGenerator(
    model="./phi-3-mini-first-responder",
    task="text-generation",
    generation_kwargs={
        "max_new_tokens": 350,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.2,
        "do_sample": True,
    }
)
```

### Flan-T5 Configuration

For Flan-T5 models:

```python
generator = HuggingFaceLocalGenerator(
    model="./flan-t5-large-first-responder",
    task="text2text-generation",
    generation_kwargs={
        "max_new_tokens": 150,
        "temperature": 0.7,
        "do_sample": True,
    }
)
```

## Hardware Optimization

### Apple Silicon Support

When running on Apple Silicon (M1/M2/M3/M4):

```python
generator = HuggingFaceLocalGenerator(
    model="./tinyllama-1.1b-first-responder-fast",  # TinyLlama works well on Apple Silicon
    task="text-generation",
    device="mps"  # Enable Metal Performance Shaders
)
```

### NVIDIA GPU Support

For NVIDIA GPUs, enable CUDA acceleration:

```python
generator = HuggingFaceLocalGenerator(
    model="./llama-3.1-1b-first-responder",  # Larger models work well with CUDA
    task="text-generation",
    device="cuda:0"
)
```

## Advanced Configuration

### Using 4-bit Quantization

For improved memory efficiency:

```python
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

generator = HuggingFaceLocalGenerator(
    model="./phi-3-mini-first-responder",
    task="text-generation",
    model_kwargs={"quantization_config": quantization_config}
)
```

### Custom Prompt Templates

Create question-specific prompts:

```python
medical_template = """
You are a medical first responder assistant. Answer the following question based on the provided context:

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Medical Question: {{ query }}

Answer (be precise and accurate):
"""

fire_template = """
You are a firefighting protocol assistant. Answer the following question based on the provided context:

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Firefighting Question: {{ query }}

Answer (focus on safety procedures):
"""

# Use different templates based on query type
def select_template(query):
    if "medical" in query.lower() or "patient" in query.lower():
        return medical_template
    elif "fire" in query.lower() or "extinguish" in query.lower():
        return fire_template
    return default_template
```

## Evaluation and Metrics

To evaluate your RAG pipeline:

```python
from haystack.components.evaluators import RougeEvaluator

evaluator = RougeEvaluator()

# Example evaluation set
eval_set = [
    {
        "query": "What is the correct rate for CPR?", 
        "answer": "The correct rate for CPR is 100-120 compressions per minute."
    },
    # Add more evaluation examples
]

results = []
for example in eval_set:
    response = pipe.run({
        "retriever": {"query": example["query"]},
        "prompt_builder": {"query": example["query"]}
    })
    generated_answer = response["llm"]["replies"][0]
    score = evaluator.run(
        predictions=[generated_answer],
        references=[example["answer"]]
    )
    results.append({"query": example["query"], "score": score})

# Calculate average scores
avg_rouge1 = sum(r["score"]["rouge1"] for r in results) / len(results)
print(f"Average ROUGE-1: {avg_rouge1}")
```

## Integrating with Web Applications

Example of setting up a simple API with FastAPI:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from haystack import Pipeline
# Import other necessary components

app = FastAPI()

# Initialize your pipeline (as shown above)
pipe = Pipeline()
# Set up the complete pipeline...

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    try:
        response = pipe.run({
            "retriever": {"query": query.question},
            "prompt_builder": {"query": query.question}
        })
        
        return {
            "answer": response["llm"]["replies"][0],
            "sources": [doc.content for doc in response["retriever"]["documents"]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Optimizations for Production

When deploying your FirstRespondersChatbot RAG system to production, consider these optimizations:

### 1. Document Store Selection

For larger document collections, consider using more scalable document stores:

```python
from haystack.document_stores.weaviate import WeaviateDocumentStore

document_store = WeaviateDocumentStore(
    url="http://localhost:8080",
    embedding_dim=384,  # Match your embedding model's dimension
    index="FirstResponders"
)
```

### 2. Batch Processing for Document Indexing

For large document collections, process in batches:

```python
# Process documents in batches of 100
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    embedded_batch = embedder.run(documents=batch)["documents"]
    document_store.write_documents(embedded_batch)
    print(f"Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
```

### 3. Caching for Frequent Queries

Implement caching for frequent queries to reduce computation:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_response(query_string):
    # Convert to immutable type for caching
    response = pipe.run({
        "retriever": {"query": query_string},
        "prompt_builder": {"query": query_string}
    })
    return response["llm"]["replies"][0]
```

### 4. Model Quantization and Serving

For TinyLlama models, leverage 4-bit quantization in production:

```python
from transformers import BitsAndBytesConfig
import torch

# 4-bit quantization with double quantization for even more memory savings
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

generator = HuggingFaceLocalGenerator(
    model="./tinyllama-1.1b-first-responder-fast",
    task="text-generation",
    model_kwargs={
        "quantization_config": quantization_config,
        "device_map": "auto"  # Automatically determine best device placement
    }
)
```

### 5. TinyLlama-Specific Template

Optimize your prompt template specifically for TinyLlama:

```python
tinyllama_template = """<s>[INST] <<SYS>>
You are a first responder assistant designed to provide accurate, concise information based on official protocols and emergency response manuals. 
Answer questions using the provided context information.
Focus on delivering complete, accurate responses that address the core purpose and function of the equipment or procedures being discussed.
<</SYS>>

Answer the following question based on the context provided:

Context: 
{% for document in documents %}
{{ document.content }}
{% endfor %}

Question: {{ query }} [/INST]
"""
```

## Conclusion

By following this guide, you can effectively integrate your trained FirstRespondersChatbot model with Haystack 2.0 to create powerful and efficient RAG pipelines. This approach combines the strengths of both retrieval and generation to provide accurate and contextual responses for first responder queries.

The TinyLlama model configuration ("tinyllama-1.1b-first-responder-fast") provides an excellent balance of speed and response quality, making it ideal for deployment in time-sensitive first responder environments. For scenarios where response quality is paramount, you can easily switch to the Llama 3.1 1B or Phi-3 Mini models by adjusting the configuration.