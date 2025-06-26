import os
import json
import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
from .rag_file_utils import FileProcessor
from .rag_chunk_tracker import ChunkTracker
from .embedding_manager import EmbeddingManager

def get_rag_logger() -> logging.Logger:
    """
    Возвращает отдельный логгер для RAG с файловым обработчиком logs/rag.log
    """
    logger = logging.getLogger("RAGRetriever")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler("logs/rag.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def save_chunks_txt(data: List[dict], filename: str = "logs/rag_chunks.txt"):
    """
    Сохраняет список чанков с их score и текстом в обычный текстовый файл (для анализа).
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a", encoding="utf-8") as f:
        for item in data:
            f.write(f"chunk_id: {item['chunk_id']}\n")
            f.write(f"score: {item['score']}\n")
            f.write(f"text: {item['text']}\n")
            f.write("-" * 40 + "\n")

class RAGRetriever:
    """
    Универсальный класс для семантического поиска по базе знаний с поддержкой автозагрузки и автосохранения
    эмбеддингов и чанков. Поддерживает эффективную работу с большими объемами данных и динамическое обновление.
    """

    def __init__(self, config: dict):
        self.logger = get_rag_logger()
        self.file_processor = FileProcessor()
        self.embed_mgr = EmbeddingManager(config.get("embedding_model", "all-MiniLM-L6-v2"))
        self.chunk_tracker = ChunkTracker()
        self.config = config

        # Пути к данным
        self.index_path = config.get("index_path", "data/faiss_index.idx")
        self.embeddings_path = config.get("embeddings_path", "data/embeddings.npy")
        self.chunks_path = config.get("chunks_path", "data/chunks.json")

        # Основные объекты
        self.chunks: List[str] = []
        self.chunk_ids: List[int] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional['faiss.Index'] = None

        # Автоматическая загрузка базы знаний
        self._autoload_knowledge_base()

    def _autoload_knowledge_base(self):
        # Автозагрузка индекса
        if os.path.exists(self.index_path):
            try:
                self.index = self.embed_mgr.load_index(self.index_path)
                self.logger.info(f"FAISS index loaded from {self.index_path}")
            except Exception as e:
                self.logger.critical(f"Failed to load FAISS index: {e}")
                self.index = None
        else:
            self.logger.warning(f"Index file not found: {self.index_path}")

        # Автозагрузка эмбеддингов
        if os.path.exists(self.embeddings_path):
            try:
                self.embeddings = np.load(self.embeddings_path)
                self.logger.info(f"Embeddings loaded from {self.embeddings_path}")
            except Exception as e:
                self.logger.critical(f"Failed to load embeddings: {e}")
                self.embeddings = None
        else:
            self.logger.warning(f"Embeddings file not found: {self.embeddings_path}")

        # Автозагрузка чанков
        if os.path.exists(self.chunks_path):
            try:
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                self.logger.info(f"Chunks loaded from {self.chunks_path}")
            except Exception as e:
                self.logger.critical(f"Failed to load chunks: {e}")
                self.chunks = []
        else:
            self.logger.warning(f"Chunks file not found: {self.chunks_path}")

        # Проверка согласованности размеров
        if self.embeddings is not None and self.chunks:
            if len(self.embeddings) != len(self.chunks):
                self.logger.error("Embeddings and chunks count mismatch. Database may be corrupted.")
                self.index = None

    def process_inform_folder(self, folder_path: str):
        """Обрабатывает папку с файлами, извлекая из них текст для построения базы знаний."""
        texts = []
        for root, _, files in os.walk(folder_path):
            for f in files:
                full = os.path.join(root, f)
                if self.file_processor.validate_file(full):
                    txt = self.file_processor.extract_text_from_file(full)
                    if txt:
                        texts.append(txt)
        chunked = []
        for t in texts:
            chunked += self.chunk_text(t, self.config.get("chunk_size", 512))
        self.chunks = chunked
        self.chunk_ids = list(range(len(chunked)))

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: Optional[int] = None) -> List[str]:
        """Разбивает текст на чанки с перекрытием."""
        overlap = overlap if overlap is not None else self.config.get("chunk_overlap", 50)
        tokens = text.split()
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk = " ".join(tokens[i:i+chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def build_knowledge_base(self):
        """Строит базу знаний заново, с сохранением всех файлов."""
        if not self.chunks:
            self.logger.error("No chunks to build knowledge base. Did you run process_inform_folder?")
            raise RuntimeError("No chunks found for building knowledge base.")
        self.embeddings = self.embed_mgr.encode_texts(self.chunks)
        self.index = self.embed_mgr.build_faiss_index(self.embeddings)
        # Сохраняем всё
        self.embed_mgr.save_index(self.index, self.index_path)
        np.save(self.embeddings_path, self.embeddings)
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        self.chunk_tracker.reset_usage_stats()
        self.logger.info("RAG knowledge base (index, embeddings, chunks) initialized and saved.")

    def retrieve_context(self, query: str, max_length: Optional[int] = None) -> str:
        """
        Получает наиболее релевантные чанки для запроса.
        Логирует запрос, id и score чанков, итоговый context.
        """
        self.logger.info(f"[RAG] Incoming query: {query!r}")
        if self.index is None or self.embeddings is None or not self.chunks:
            self.logger.error("Knowledge base is not loaded. Please build or load the knowledge base first.")
            raise RuntimeError("Knowledge base not loaded! Run build_knowledge_base or check data files.")

        q_emb = self.embed_mgr.encode_single_text(query)
        ids, sims = self.embed_mgr.search_similar(self.index, q_emb, k=20)

        # Защита от выхода за границы
        candidate_chunks = [(int(i), self.chunks[int(i)]) for i in ids if int(i) < len(self.chunks)]
        candidate_info = [
            {
                "chunk_id": int(i),
                "score": float(sims[idx]),
                "text": self.chunks[int(i)]
            }
            for idx, i in enumerate(ids) if int(i) < len(self.chunks)
        ]
        # Сохраняем кандидатов в обычный txt
        save_chunks_txt(candidate_info, filename="logs/rag_chunks.txt")

        diverse = self.chunk_tracker.get_diverse_chunks(candidate_chunks)
        selected = diverse[:self.config.get("max_context_chunks", 5)]

        selection_info = []
        for order, (chunk_id, chunk_text) in enumerate(selected, 1):
            pos = list(ids).index(chunk_id) if chunk_id in ids else -1
            score = float(sims[pos]) if pos != -1 else float("nan")
            selection_info.append({
                "selected_order": int(order),
                "chunk_id": int(chunk_id),
                "score": float(score),
                "text": chunk_text
            })
            self.chunk_tracker.track_usage(chunk_id=int(chunk_id), topic=query)

        # Сохраняем выбранные чанки в отдельный txt
        save_chunks_txt(selection_info, filename="logs/rag_selected_chunks.txt")

        context = "\n\n".join([chunk for _, chunk in selected])
        self.logger.info(f"[RAG] Query: {query!r}\nSelected chunks count: {len(selection_info)}")
        self.logger.debug(f"[RAG] Final context for query: {query!r}\n{context[:2000]}")

        return context

    def update_knowledge_base(self, new_content: str, source: str = None):
        """Добавляет новые чанки и эмбеддинги в существующую базу знаний."""
        if self.index is None or self.embeddings is None or not self.chunks:
            self.logger.warning("Knowledge base not loaded. Rebuilding from scratch.")
            self.chunks = []
            self.embeddings = None
            self.index = None
        new_chunks = self.chunk_text(new_content, self.config.get("chunk_size", 512))
        new_embs = self.embed_mgr.encode_texts(new_chunks)
        self.chunks.extend(new_chunks)
        if self.embeddings is not None:
            self.embeddings = np.vstack([self.embeddings, new_embs])
        else:
            self.embeddings = new_embs
        if self.index is not None:
            self.index.add(new_embs)
        else:
            self.index = self.embed_mgr.build_faiss_index(self.embeddings)
        # Сохраняем обновления
        self.embed_mgr.save_index(self.index, self.index_path)
        np.save(self.embeddings_path, self.embeddings)
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        start_id = len(self.chunks) - len(new_chunks)
        for i in range(len(new_chunks)):
            self.chunk_tracker.track_usage(chunk_id=start_id + i, topic=source or "update")
        self.logger.info(f"Knowledge base updated with {len(new_chunks)} new chunks.")

    def get_relevant_chunks(self, topic: str, limit: int = 10) -> List[str]:
        """Возвращает релевантные чанки по теме."""
        context = self.retrieve_context(topic)
        return context.split("\n\n")

    def get_index_stats(self) -> dict:
        """Статистика базы знаний."""
        return {
            "total_chunks": len(self.chunks),
            "index_loaded": self.index is not None,
            "embeddings_loaded": self.embeddings is not None
        }
