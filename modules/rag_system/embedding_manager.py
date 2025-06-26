import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class EmbeddingManager:
    """
    Менеджер для работы с эмбеддингами и faiss-индексом для поиска схожих текстов.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.logger = logging.getLogger("EmbeddingManager")
        self.model = SentenceTransformer(model_name)
        self.logger.info(f"Embedding model loaded: {model_name}")

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Векторизация батча текстов."""
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return np.array(embeddings, dtype='float32')

    def encode_single_text(self, text: str) -> np.ndarray:
        """Векторизация одного текста."""
        return np.array(self.model.encode(text), dtype='float32')

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Строит FAISS индекс по эмбеддингам."""
        if embeddings.ndim != 2:
            raise ValueError("Embeddings array must be 2-dimensional")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.logger.info(f"FAISS index built with {embeddings.shape[0]} vectors of dim={dim}")
        return index

    def save_index(self, index: faiss.Index, path: str) -> None:
        """Сохраняет faiss-индекс на диск."""
        faiss.write_index(index, path)
        self.logger.info(f"FAISS index saved to {path}")

    def load_index(self, path: str) -> faiss.Index:
        """Загружает faiss-индекс с диска."""
        return faiss.read_index(path)

    def search_similar(self, index: Optional[faiss.Index], query_emb: np.ndarray, k: int = 5):
        """Выполняет поиск наиболее похожих эмбеддингов в индексе."""
        if index is None:
            raise RuntimeError("FAISS index is None. Проверьте, загружен ли индекс.")
        D, I = index.search(np.array([query_emb]), k)
        return I[0], D[0]
