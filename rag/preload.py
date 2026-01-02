from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

print("Pre-downloading BGE-M3 Embedding Model to cache...")
# This forces the download to the HF_HOME cache directory
model = HuggingFaceEmbedding(model_name="BAAI/bge-large-zh-v1.5")
print("Model downloaded successfully.")