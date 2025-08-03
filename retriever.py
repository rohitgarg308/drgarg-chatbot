# retrieve top-matching vector data
import os
import openai
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Paths
VECTOR_DIR = "vectorstore"
INDEX_FILE = os.path.join(VECTOR_DIR, "index.faiss")
DOCS_FILE = os.path.join(VECTOR_DIR, "docs.pkl")

# Load index and docs
index = faiss.read_index(INDEX_FILE)
with open(DOCS_FILE, "rb") as f:
  documents = pickle.load(f)


# Main retriever function
def get_best_match(query, top_k=1):
  # Create embedding for the query
  response = openai.embeddings.create(input=[query],
                                      model="text-embedding-ada-002")
  query_vector = np.array(response.data[0].embedding).astype("float32")

  # Search in FAISS index
  D, I = index.search(np.array([query_vector]), top_k)
  matches = [documents[i] for i in I[0] if i < len(documents)]

  return matches
