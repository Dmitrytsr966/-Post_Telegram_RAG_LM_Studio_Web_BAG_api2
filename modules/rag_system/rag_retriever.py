import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

from .rag_file_utils import FileProcessor
from .rag_chunk_tracker import ChunkTracker
from .embedding_manager import EmbeddingManager

# ================================ #
# 1. Фильтрация и Аналитика Чанков #
# ================================ #

def is_trash_chunk(chunk: str) -> bool:
    """
    Возвращает True, если чанк пустой, состоит только из спецсимволов/пробелов или html-тегов.
    Причина: Защита от попадания мусора в базу и выдачу.
    Следствие: Повышение качества базы и выдачи.
    """
    import re
    text = re.sub(r'<.*?>', '', chunk)
    text = text.strip()
    if not text or len(text) < 5:
        return True
    if re.fullmatch(r'[\W_]+', text):
        return True
    return False

def filter_trash_chunks(chunks: List[str]) -> List[str]:
    """
    Удаляет мусорные чанки.
    Причина: Повышение качества базы.
    """
    return [ch for ch in chunks if not is_trash_chunk(ch)]

def analyze_trash_chunks(chunks: List[str]) -> Dict[str, Any]:
    """
    Аналитика по чанкам: процент мусорных, примеры.
    Причина: Анализ качества данных.
    """
    trash = [ch for ch in chunks if is_trash_chunk(ch)]
    percent = len(trash) / len(chunks) * 100 if chunks else 0
    return {
        'total': len(chunks),
        'trash_count': len(trash),
        'trash_percent': percent,
        'trash_examples': trash[:5]
    }

# ===================== #
# 2. Логгер и Мониторинг #
# ===================== #

def get_rag_logger() -> logging.Logger:
    """
    Возвращает отдельный логгер для RAG с файловым обработчиком logs/rag.log
    Причина: Централизованный контроль логов.
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
    Причина: Диагностика и ручной аудит.
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "a", encoding="utf-8") as f:
            for item in data:
                f.write(f"chunk_id: {item.get('chunk_id')}\n")
                if "selected_order" in item:
                    f.write(f"selected_order: {item['selected_order']}\n")
                f.write(f"score: {item.get('score')}\n")
                txt = item.get('text')
                if isinstance(txt, str):
                    txt = txt.replace('\n', ' ')[:1000]
                f.write(f"text: {txt}\n")
                f.write("-" * 40 + "\n")
    except Exception as e:
        logging.getLogger("RAGRetriever").error(f"Failed to save chunks log '{filename}': {e}")

def log_duplicates(duplicates: List[Dict[str, Any]], filename="logs/rag_duplicates.txt"):
    """
    Логирует найденные дубликаты чанков для аудита.
    """
    if not duplicates:
        return
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "a", encoding="utf-8") as f:
            for entry in duplicates:
                f.write(f"Duplicate chunk_id: {entry['chunk_id']}\nText: {entry['text'][:200]}\n{'-'*40}\n")
    except Exception as e:
        logging.getLogger("RAGRetriever").error(f"Failed to log duplicates: {e}")

# ==================================== #
# 3. RAGRetriever: Контракт и Архитектура #
# ==================================== #

class RAGRetriever:
    """
    SOLID: 
      - Single Responsibility: Только управление поиском/индексацией/выдачей.
      - Dependency Injection через config.
      - Интерфейс стабильный: все методы с чёткими контрактами.
    """

    def __init__(self, config: dict):
        """
        Инициализация.
        - Загружает конфиги, инициализирует зависимости, автоматически поднимает базу знаний.
        Вход: config (dict)
        """
        self.logger = get_rag_logger()
        self.file_processor = FileProcessor()
        self.embed_mgr = EmbeddingManager(config.get("embedding_model", "all-MiniLM-L6-v2"))
        self.chunk_tracker = ChunkTracker()
        self.config = config

        self.index_path = config.get("index_path", "data/faiss_index.idx")
        self.embeddings_path = config.get("embeddings_path", "data/embeddings.npy")
        self.chunks_path = config.get("chunks_path", "data/chunks.json")
        self.chunks: List[str] = []
        self.chunk_ids: List[int] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[Any] = None

        self._autoload_knowledge_base()

    # ========== Автоматическая загрузка базы знаний ==========

    def _autoload_knowledge_base(self):
        """
        Загружает индекс, эмбеддинги, чанки. Проверяет согласованность.
        Причина: Автоматизация старта, fail-fast при ошибке.
        """
        # Индекс
        if os.path.exists(self.index_path):
            try:
                self.index = self.embed_mgr.load_index(self.index_path)
                self.logger.info(f"FAISS index loaded from {self.index_path}")
            except Exception as e:
                self.logger.critical(f"Failed to load FAISS index: {e}", exc_info=True)
                self.index = None
        else:
            self.logger.warning(f"Index file not found: {self.index_path}")
        # Эмбеддинги
        if os.path.exists(self.embeddings_path):
            try:
                self.embeddings = np.load(self.embeddings_path)
                self.logger.info(f"Embeddings loaded from {self.embeddings_path}")
            except Exception as e:
                self.logger.critical(f"Failed to load embeddings: {e}", exc_info=True)
                self.embeddings = None
        else:
            self.logger.warning(f"Embeddings file not found: {self.embeddings_path}")
        # Чанки
        if os.path.exists(self.chunks_path):
            try:
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                self.logger.info(f"Chunks loaded from {self.chunks_path}")
            except Exception as e:
                self.logger.critical(f"Failed to load chunks: {e}", exc_info=True)
                self.chunks = []
        else:
            self.logger.warning(f"Chunks file not found: {self.chunks_path}")
        # Согласованность
        if self.embeddings is not None and self.chunks:
            if len(self.embeddings) != len(self.chunks):
                self.logger.error("Embeddings and chunks count mismatch. Database may be corrupted.")
                self.index = None
                self.embeddings = None
                self.chunks = []
                raise RuntimeError("Knowledge base is corrupted (embedding/chunk size mismatch)!")

    # ========== Обработка папки inform ==========

    def process_inform_folder(self, folder_path: str):
        """
        Сканирует папку, извлекает и фильтрует текст, разбивает на чанки.
        Вход: folder_path (str)
        Выход: None (заполняет self.chunks, self.chunk_ids)
        Edge cases: битые/большие/пустые файлы; мусорные чанки.
        """
        texts = []
        for root, _, files in os.walk(folder_path):
            for f in files:
                full = os.path.join(root, f)
                try:
                    if self.file_processor.validate_file(full):
                        txt = self.file_processor.extract_text_from_file(full)
                        if txt:
                            texts.append(txt)
                except Exception as e:
                    self.logger.warning(f"Failed to process file {full}: {e}", exc_info=True)
        chunked = []
        for t in texts:
            raw_chunks = self.chunk_text(t, self.config.get("chunk_size", 512))
            filtered_chunks = filter_trash_chunks(raw_chunks)
            chunked += filtered_chunks
            stats = analyze_trash_chunks(raw_chunks)
            self.logger.info(f"File chunk stats: {stats}")
        self.chunks = chunked
        self.chunk_ids = list(range(len(chunked)))
        self.logger.info(f"Total clean chunks after filtering: {len(chunked)}")

    # ========== Чанкинг ==========

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: Optional[int] = None) -> List[str]:
        """
        Разбивает текст на чанки с перекрытием.
        Причина: Унификация входных данных.
        Edge: chunk_size=0, overlap >= chunk_size, пустой текст.
        """
        if not text or chunk_size <= 0:
            return []
        overlap = overlap if overlap is not None else self.config.get("chunk_overlap", 50)
        tokens = text.split()
        step = max(chunk_size - overlap, 1)
        chunks = []
        for i in range(0, len(tokens), step):
            chunk = " ".join(tokens[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    # ========== Построение базы знаний ==========

    def build_knowledge_base(self):
        """
        Строит и сохраняет базу знаний.
        Контракт: все компоненты базы должны быть согласованы по размеру.
        """
        if not self.chunks:
            self.logger.error("No chunks to build knowledge base. Did you run process_inform_folder?")
            raise RuntimeError("No chunks found for building knowledge base.")
        clean_chunks = filter_trash_chunks(self.chunks)
        trash_stats = analyze_trash_chunks(self.chunks)
        self.logger.info(f"Build KB: {trash_stats}")
        if not clean_chunks:
            self.logger.error("All chunks are trash after filtering!")
            raise RuntimeError("All chunks are trash after filtering!")
        self.chunks = clean_chunks
        self.chunk_ids = list(range(len(self.chunks)))
        try:
            self.embeddings = self.embed_mgr.encode_texts(self.chunks)
            self.index = self.embed_mgr.build_faiss_index(self.embeddings)
        except Exception as e:
            self.logger.critical(f"Failed to build embeddings/index: {e}", exc_info=True)
            raise
        try:
            self.embed_mgr.save_index(self.index, self.index_path)
            np.save(self.embeddings_path, self.embeddings)
            with open(self.chunks_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.critical(f"Failed to save KB files: {e}", exc_info=True)
            raise
        self.chunk_tracker.reset_usage_stats()
        self.logger.info("RAG knowledge base (index, embeddings, chunks) initialized and saved.")

    # ========== Поиск контекста (Улучшено: фильтрация дубликатов, алерт логов) ==========

    def retrieve_context(self, query: str, max_length: Optional[int] = None) -> str:
        """
        Получает наиболее релевантные чанки для запроса.
        Контракт: всегда возвращает не-дублирующийся контекст.
        """
        self.logger.info(f"[RAG] Incoming query: {query!r}")
        if self.index is None or self.embeddings is None or not self.chunks:
            self.logger.error("Knowledge base is not loaded. Please build or load the knowledge base first.")
            raise RuntimeError("Knowledge base not loaded! Run build_knowledge_base or check data files.")
        try:
            q_emb = self.embed_mgr.encode_single_text(query)
            ids, sims = self.embed_mgr.search_similar(self.index, q_emb, k=20)
        except Exception as e:
            self.logger.critical(f"Failed to encode/search query '{query}': {e}", exc_info=True)
            raise
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
        try:
            save_chunks_txt(candidate_info, filename="logs/rag_chunks.txt")
        except Exception as e:
            self.logger.error(f"Failed to write rag_chunks.txt: {e}", exc_info=True)
        # Фильтрация мусорных
        filtered_candidates = [(cid, txt) for cid, txt in candidate_chunks if not is_trash_chunk(txt)]
        trash_candidate_count = len(candidate_chunks) - len(filtered_candidates)
        self.logger.info(f"Trash candidates: {trash_candidate_count}/{len(candidate_chunks)}")
        diverse = self.chunk_tracker.get_diverse_chunks(filtered_candidates)
        # Новый шаг: фильтрация дубликатов
        seen_chunks = set()
        unique_selected = []
        duplicates_log = []
        for cid, chunk in diverse:
            if chunk not in seen_chunks:
                unique_selected.append((cid, chunk))
                seen_chunks.add(chunk)
            else:
                duplicates_log.append({"chunk_id": cid, "text": chunk})
            if len(unique_selected) >= self.config.get("max_context_chunks", 5):
                break
        if duplicates_log:
            log_duplicates(duplicates_log)
            self.logger.info(f"Detected {len(duplicates_log)} duplicate chunks in context.")
        # Ограничиваем длину контекста по max_length (если указано)
        context_chunks = []
        total_length = 0
        for _, chunk in unique_selected:
            if max_length is not None and total_length + len(chunk) > max_length:
                break
            context_chunks.append(chunk)
            total_length += len(chunk)
        selection_info = []
        for order, (chunk_id, chunk_text) in enumerate(unique_selected[:len(context_chunks)], 1):
            try:
                pos = list(ids).index(chunk_id) if chunk_id in ids else -1
                score = float(sims[pos]) if pos != -1 else float("nan")
            except Exception:
                score = float("nan")
            selection_info.append({
                "selected_order": int(order),
                "chunk_id": int(chunk_id),
                "score": float(score),
                "text": chunk_text
            })
            self.chunk_tracker.track_usage(chunk_id=int(chunk_id), topic=query)
        try:
            save_chunks_txt(selection_info, filename="logs/rag_selected_chunks.txt")
        except Exception as e:
            self.logger.error(f"Failed to write rag_selected_chunks.txt: {e}", exc_info=True)
        context = "\n\n".join(context_chunks)
        self.logger.info(f"[RAG] Query: {query!r}\nSelected unique chunks count: {len(selection_info)}")
        self.logger.debug(f"[RAG] Final context for query: {query!r}\n{context[:2000]}")
        return context

    # ========== Добавление новых чанков ==========

    def update_knowledge_base(self, new_content: str, source: str = None):
        """
        Добавляет новые чанки и эмбеддинги в существующую базу знаний.
        Контракт: база и индекс согласованы.
        """
        if self.index is None or self.embeddings is None or not self.chunks:
            self.logger.warning("Knowledge base not loaded. Rebuilding from scratch.")
            self.chunks = []
            self.embeddings = None
            self.index = None
        raw_chunks = self.chunk_text(new_content, self.config.get("chunk_size", 512))
        new_chunks = filter_trash_chunks(raw_chunks)
        stats = analyze_trash_chunks(raw_chunks)
        self.logger.info(f"Update KB: {stats}")
        existing_set = set(self.chunks)
        unique_new_chunks = [ch for ch in new_chunks if ch not in existing_set]
        if not unique_new_chunks:
            self.logger.info("No unique new chunks to add.")
            return
        try:
            new_embs = self.embed_mgr.encode_texts(unique_new_chunks)
        except Exception as e:
            self.logger.critical(f"Failed to encode new content: {e}", exc_info=True)
            raise
        self.chunks.extend(unique_new_chunks)
        if self.embeddings is not None:
            self.embeddings = np.vstack([self.embeddings, new_embs])
        else:
            self.embeddings = new_embs
        if self.index is not None:
            self.index.add(new_embs)
        else:
            self.index = self.embed_mgr.build_faiss_index(self.embeddings)
        try:
            self.embed_mgr.save_index(self.index, self.index_path)
            np.save(self.embeddings_path, self.embeddings)
            with open(self.chunks_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.critical(f"Failed to save KB files after update: {e}", exc_info=True)
            raise
        start_id = len(self.chunks) - len(unique_new_chunks)
        for i in range(len(unique_new_chunks)):
            self.chunk_tracker.track_usage(chunk_id=start_id + i, topic=source or "update")
        self.logger.info(f"Knowledge base updated with {len(unique_new_chunks)} new unique chunks.")

    # ========== Сервисные методы ==========

    def get_relevant_chunks(self, topic: str, limit: int = 10) -> List[str]:
        """
        Возвращает релевантные чанки по теме, не более limit.
        """
        context = self.retrieve_context(topic, max_length=None)
        chunks = context.split("\n\n")
        return chunks[:limit]

    def get_index_stats(self) -> dict:
        """
        Статистика базы знаний.
        """
        return {
            "total_chunks": len(self.chunks),
            "index_loaded": self.index is not None,
            "embeddings_loaded": self.embeddings is not None
        }
