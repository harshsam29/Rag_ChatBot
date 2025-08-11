from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from loguru import logger

class Retriever:
    def __init__(self, index_path, chunks_path):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.index = faiss.read_index(index_path)
            with open(chunks_path, 'rb') as f:
                self.data = pickle.load(f)
            self.chunks = [c["text"] for c in self.data["chunks"]]
            self.metadata = [c["metadata"] for c in self.data["chunks"]]
            logger.info(f"Loaded {len(self.chunks)} chunks from {chunks_path}")
        except Exception as e:
            logger.error(f"Error initializing Retriever: {e}")
            raise

    def retrieve(self, query, k=5):
        try:
            query_embedding = self.model.encode([query])
            distances, indices = self.index.search(np.array(query_embedding), k)
            retrieved_chunks = [(self.chunks[idx], self.metadata[idx]) for idx in indices[0]]
            return retrieved_chunks, distances[0]
        except Exception as e:
            logger.error(f"Error retrieving chunks for query '{query}': {e}")
            raise