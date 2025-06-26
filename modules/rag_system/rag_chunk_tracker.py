import logging
import json
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta

class ChunkTracker:
    """
    Трекинг использования чанков знаний для разнообразия, аналитики и penalty-функций.
    """
    def __init__(self, logger: logging.Logger = None):
        self.usage: Dict[str, List[Dict[str, Any]]] = {}  # chunk_id -> list of usage dicts
        self.logger = logger or logging.getLogger("ChunkTracker")

    def track_usage(self, chunk_id: str, topic: str, dt: datetime = None):
        """Сохраняет факт использования чанка для темы."""
        entry = {
            "topic": topic,
            "timestamp": (dt or datetime.utcnow()).isoformat()
        }
        self.usage.setdefault(str(chunk_id), []).append(entry)
        self.logger.debug(f"Tracked usage: chunk_id={chunk_id}, topic={topic}")

    def get_usage_penalty(self, chunk_id: str) -> float:
        """Возвращает penalty за частое использование чанка (например, просто длина истории)."""
        return float(len(self.usage.get(str(chunk_id), [])))

    def get_usage_count(self, chunk_id: str) -> int:
        """Сколько раз этот чанк уже использовался."""
        return len(self.usage.get(str(chunk_id), []))

    def reset_usage_stats(self):
        """Сброс всего трекинга (например, при перестроении базы знаний)."""
        self.usage = {}
        self.logger.info("Chunk usage stats reset.")

    def get_diverse_chunks(self, candidates: List[Tuple[int, str]], limit: int = None) -> List[Tuple[int, str]]:
        """
        candidates: список (chunk_id, chunk_text)
        Возвращает чанки, сортируя по минимальному использованию (diversity).
        Если задан limit — обрезает список до limit.
        """
        sorted_chunks = sorted(
            candidates,
            key=lambda x: (self.get_usage_count(x[0]), x[0])
        )
        result = sorted_chunks[:limit] if limit is not None else sorted_chunks
        self.logger.debug(f"Selected diverse chunks: {[c[0] for c in result]}")
        return result

    def apply_penalty_scores(self, chunks: List[Tuple[int, str]]) -> List[Tuple[int, str, float]]:
        """
        Возвращает те же чанки, но с добавленной penalty-оценкой (для внутренней сортировки).
        """
        scored = [(chunk_id, chunk, self.get_usage_penalty(chunk_id)) for chunk_id, chunk in chunks]
        self.logger.debug("Applied penalty scores to chunks.")
        return scored

    def get_tracker_stats(self) -> dict:
        """Возвращает агрегированную статистику по использованию чанков."""
        stats = {
            "total_tracked": len(self.usage),
            "usage_counts": {k: len(v) for k, v in self.usage.items()}
        }
        self.logger.info(f"ChunkTracker stats: {stats}")
        return stats

    def save_usage_data(self, file_path: str):
        """Сохраняет usage-статистику в файл (json)."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.usage, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Chunk usage data saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save usage data: {e}")

    def load_usage_data(self, file_path: str):
        """Загружает usage-статистику из файла (json)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.usage = json.load(f)
            self.logger.info(f"Chunk usage data loaded from {file_path}")
        except FileNotFoundError:
            self.usage = {}
            self.logger.warning(f"Usage data file not found: {file_path}, starting fresh.")
        except Exception as e:
            self.logger.error(f"Failed to load usage data: {e}")

    def cleanup_old_usage(self, days_threshold: int = 30):
        """Очищает usage-логи старше заданного количества дней."""
        cutoff = datetime.utcnow() - timedelta(days=days_threshold)
        cutoff_iso = cutoff.isoformat()
        removed = 0
        for chunk_id, usage_list in list(self.usage.items()):
            new_list = [entry for entry in usage_list if entry.get("timestamp", "") > cutoff_iso]
            removed += len(usage_list) - len(new_list)
            self.usage[chunk_id] = new_list
        self.logger.info(f"Old usage cleaned: {removed} entries removed (older than {days_threshold} days).")
