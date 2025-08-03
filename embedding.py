import os
import openai
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv

# Load API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Directories
DATA_DIR = "data"
VECTOR_DIR = "vectorstore"
INDEX_FILE = os.path.join(VECTOR_DIR, "index.faiss")
DOCS_FILE = os.path.join(VECTOR_DIR, "docs.pkl")

# Load and split data
documents = []
for filename in os.listdir(DATA_DIR):
  with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
    content = f.read()
    documents.append((filename, content))

# Prepare texts
texts = [doc[1] for doc in documents]

# Create embeddings using new OpenAI client
response = openai.embeddings.create(input=texts,
                                    model="text-embedding-ada-002")
vectors = [r.embedding for r in response.data]

# Build FAISS index
dimension = len(vectors[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors).astype("float32"))

# Save vector index and document mapping
os.makedirs(VECTOR_DIR, exist_ok=True)
faiss.write_index(index, INDEX_FILE)
with open(DOCS_FILE, "wb") as f:
  pickle.dump(documents, f)

print("✅ Embeddings created and saved!")
