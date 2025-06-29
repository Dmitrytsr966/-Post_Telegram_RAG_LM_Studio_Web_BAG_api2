import logging
import json
from typing import List, Tuple, Dict, Any, Optional, Callable
from datetime import datetime, timedelta

class ChunkTracker:
    """
    Трекинг использования чанков знаний для разнообразия, аналитики, penalty-функций и интеграции с анализом качества.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.usage: Dict[str, List[Dict[str, Any]]] = {}  # chunk_id -> list of usage dicts
        self.logger = logger or logging.getLogger("ChunkTracker")
        self.chunk_quality_fn: Optional[Callable[[str], bool]] = None

    # === Интеграция с анализом качества чанков ===

    def set_chunk_quality_checker(self, fn: Callable[[str], bool]) -> None:
        """
        Установить функцию, определяющую мусорность чанка.
        Принимает chunk_text, возвращает True если чанк мусорный.
        """
        self.chunk_quality_fn = fn
        self.logger.info("Chunk quality checker function set.")

    # === Трекинг использования ===

    def track_usage(self, chunk_id: str, topic: str, dt: Optional[datetime] = None) -> None:
        """Сохраняет факт использования чанка для темы."""
        entry = {
            "topic": topic,
            "timestamp": (dt or datetime.utcnow()).isoformat()
        }
        self.usage.setdefault(str(chunk_id), []).append(entry)
        self.logger.debug(f"Tracked usage: chunk_id={chunk_id}, topic={topic}")

    def get_usage_penalty(self, chunk_id: str) -> float:
        """Penalty за частое использование чанка (количество использований)."""
        return float(len(self.usage.get(str(chunk_id), [])))

    def get_usage_count(self, chunk_id: str) -> int:
        """Сколько раз этот чанк уже использовался."""
        return len(self.usage.get(str(chunk_id), []))

    def reset_usage_stats(self) -> None:
        """Сброс всего трекинга (например, при перестроении базы знаний)."""
        self.usage = {}
        self.logger.info("Chunk usage stats reset.")

    # === Диверсификация и Penalty ===

    def get_diverse_chunks(self, candidates: List[Tuple[int, str]], limit: Optional[int] = None) -> List[Tuple[int, str]]:
        """
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

    # === Аналитика и отчеты ===

    def get_tracker_stats(self) -> dict:
        """Возвращает агрегированную статистику по использованию чанков."""
        stats = {
            "total_tracked": len(self.usage),
            "usage_counts": {k: len(v) for k, v in self.usage.items()}
        }
        self.logger.info(f"ChunkTracker stats: {stats}")
        return stats

    def get_most_used_chunks(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Возвращает топ-N самых часто используемых чанков (chunk_id, count).
        """
        counts = [(chunk_id, len(usages)) for chunk_id, usages in self.usage.items()]
        most_used = sorted(counts, key=lambda x: x[1], reverse=True)[:top_n]
        self.logger.info(f"Top {top_n} most used chunks: {most_used}")
        return most_used

    def analyze_trash_chunks_usage(
        self, 
        chunk_id_to_text: Dict[str, str], 
        is_trash_fn: Optional[Callable[[str], bool]] = None, 
        min_usage: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Возвращает список чанков, которые часто выбирались, но являются мусорными.
        Требуется словарь chunk_id -> текст чанка.
        """
        if is_trash_fn is None:
            is_trash_fn = self.chunk_quality_fn
        if is_trash_fn is None:
            self.logger.warning("No trash-check function provided for analyze_trash_chunks_usage.")
            return []

        result = []
        for chunk_id, usages in self.usage.items():
            text = chunk_id_to_text.get(str(chunk_id), "")
            try:
                if is_trash_fn(text) and len(usages) >= min_usage:
                    result.append({
                        "chunk_id": chunk_id,
                        "usage_count": len(usages),
                        "text": text[:500]
                    })
            except Exception as e:
                self.logger.error(f"Error in trash check for chunk_id={chunk_id}: {e}")
        if result:
            self.logger.info(f"Found {len(result)} trash chunks with high usage: {[r['chunk_id'] for r in result]}")
        return result

    def crosslink_usage_with_quality(
        self, 
        chunk_id_to_text: Dict[str, str], 
        is_trash_fn: Optional[Callable[[str], bool]] = None, 
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Для аналитики: возвращает топ-N чанков по использованию с пометкой "мусорный"/"нормальный".
        """
        if is_trash_fn is None:
            is_trash_fn = self.chunk_quality_fn
        if is_trash_fn is None:
            self.logger.warning("No trash-check function provided for crosslink.")
            return []
        most_used = self.get_most_used_chunks(top_n)
        result = []
        for chunk_id, count in most_used:
            text = chunk_id_to_text.get(str(chunk_id), "")
            try:
                is_trash = bool(is_trash_fn(text))
            except Exception as e:
                self.logger.error(f"Error in trash check for chunk_id={chunk_id}: {e}")
                is_trash = False
            result.append({
                "chunk_id": chunk_id,
                "usage_count": count,
                "is_trash": is_trash,
                "text": text[:500]
            })
        self.logger.info(f"Crosslinked usage and quality for {len(result)} chunks.")
        return result

    # === Очистка статистики ===

    def remove_usage_for_chunks(self, chunk_ids: List[str]) -> None:
        """
        Удаляет usage-логи для переданных чанков (например, признанных мусорными или удалённых из базы).
        """
        removed = 0
        for cid in chunk_ids:
            if cid in self.usage:
                removed += len(self.usage[cid])
                del self.usage[cid]
        self.logger.info(f"Removed usage logs for {len(chunk_ids)} chunks, total {removed} entries.")

    def remove_usage_for_missing_chunks(self, actual_chunk_ids: Optional[set] = None) -> None:
        """
        Удаляет usage-логи для чанков, которых нет в базе знаний.
        :param actual_chunk_ids: множество chunk_id, которые существуют в базе знаний.
        """
        if actual_chunk_ids is None:
            self.logger.warning("No actual_chunk_ids provided to remove_usage_for_missing_chunks.")
            return
        to_remove = [cid for cid in self.usage if cid not in actual_chunk_ids]
        self.remove_usage_for_chunks(to_remove)
        self.logger.info(f"Cleaned up usage logs for missing chunks: {to_remove}")

    def cleanup_old_usage(self, days_threshold: int = 30) -> None:
        """Очищает usage-логи старше заданного количества дней."""
        cutoff = datetime.utcnow() - timedelta(days=days_threshold)
        cutoff_iso = cutoff.isoformat()
        removed = 0
        for chunk_id, usage_list in list(self.usage.items()):
            new_list = []
            for entry in usage_list:
                ts = entry.get("timestamp", "")
                # Robust date comparison
                try:
                    entry_dt = datetime.fromisoformat(ts)
                except Exception:
                    entry_dt = None
                if entry_dt is not None and entry_dt > cutoff:
                    new_list.append(entry)
            removed += len(usage_list) - len(new_list)
            self.usage[chunk_id] = new_list
        self.logger.info(f"Old usage cleaned: {removed} entries removed (older than {days_threshold} days).")

    # === Персистентность ===

    def save_usage_data(self, file_path: str) -> None:
        """Сохраняет usage-статистику в файл (json)."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.usage, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Chunk usage data saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save usage data: {e}")

    def load_usage_data(self, file_path: str) -> None:
        """Загружает usage-статистику из файла (json)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.usage = json.load(f)
            self.logger.info(f"Chunk usage data loaded from {file_path}")
        except FileNotFoundError:
            self.usage = {}
            self.logger.warning(f"Usage data file not found: {file_path}, starting fresh.")
        except Exception as e:
            self.usage = {}
            self.logger.error(f"Failed to load usage data: {e}")
