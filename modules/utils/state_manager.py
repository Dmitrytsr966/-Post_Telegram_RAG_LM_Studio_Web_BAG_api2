import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union

class StateManager:
    """
    Класс для управления состоянием системы: трекинг тем, статистики,
    резервное копирование, восстановление, обработка ошибок чтения/записи состояния.
    """

    def __init__(self, state_file: Union[str, Path] = "data/state.json"):
        self.state_file = Path(state_file)
        self.logger = logging.getLogger("StateManager")

        self.default_state = {
            "processed": {},
            "unprocessed": [],
            "failed": {},
            "statistics": {
                "topics_processed": 0,
                "success_count": 0,
                "error_count": 0,
                "start_time": datetime.utcnow().isoformat()
            },
            "system_status": "INIT",
            "last_activity": datetime.utcnow().isoformat()
        }
        self.state: Dict[str, Any] = {}

        self.load_state()

    # ----------- Группа: Загрузка/сохранение состояния -----------
    def load_state(self) -> None:
        """
        Загружает состояние из файла (если он есть и валиден).
        В случае ошибки или пустого файла - инициализирует дефолтное состояние.
        """
        if not self.state_file.exists() or self.state_file.stat().st_size == 0:
            self.logger.warning(f"State file {self.state_file} missing or empty. Initializing default state.")
            self.state = self.default_state.copy()
            self.save_state()
            return
        try:
            with self.state_file.open("r", encoding="utf-8") as f:
                self.state = json.load(f)
            # Проверка обязательных полей и "дополнение" до актуальной структуры
            self._ensure_state_integrity()
            self.logger.info(f"State loaded from {self.state_file}")
        except Exception as e:
            self.logger.error("Failed to load state. Initializing default state.", exc_info=True)
            self.state = self.default_state.copy()
            self.save_state()

    def save_state(self) -> None:
        """
        Сохраняет текущее состояние в файл. Атомарно (сначала tmp, затем заменяет основной).
        """
        tmp_file = self.state_file.with_suffix(".tmp")
        try:
            with tmp_file.open("w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=4, ensure_ascii=False)
            tmp_file.replace(self.state_file)
            self.logger.debug(f"State saved to {self.state_file}")
        except Exception as e:
            self.logger.error(f"Failed to save state to {self.state_file}", exc_info=True)

    def _ensure_state_integrity(self) -> None:
        """
        Проверяет, что все обязательные поля есть в self.state (для обратной совместимости).
        Если чего-то не хватает - добавляет из default_state.
        """
        for key, value in self.default_state.items():
            if key not in self.state:
                self.state[key] = value if not isinstance(value, dict) else value.copy()
        # Для вложенных статистик
        for k, v in self.default_state["statistics"].items():
            if "statistics" not in self.state:
                self.state["statistics"] = {}
            if k not in self.state["statistics"]:
                self.state["statistics"][k] = v

    # ----------- Группа: Методы для работы с темами -----------
    def mark_topic_processed(self, topic: str, success: bool, details: Optional[dict] = None) -> None:
        """
        Отмечает тему как успешно обработанную или failed.
        Перемещает из unprocessed в processed/failed, обновляет статистику.
        """
        now = datetime.utcnow().isoformat()
        details = details or {}

        if success:
            self.state["processed"][topic] = {
                "status": "success",
                "timestamp": now,
                "details": details
            }
            self.state["statistics"]["success_count"] += 1
        else:
            self.state["failed"][topic] = {
                "status": "failed",
                "timestamp": now,
                "error": details
            }
            self.state["statistics"]["error_count"] += 1

        if topic in self.state["unprocessed"]:
            self.state["unprocessed"].remove(topic)

        self.state["statistics"]["topics_processed"] += 1
        self.state["last_activity"] = now
        self.save_state()

    def add_topic(self, topic: str) -> None:
        """
        Добавляет новую тему в список необработанных, если она ещё не встречалась.
        """
        if topic not in self.state["unprocessed"] \
           and topic not in self.state["processed"] \
           and topic not in self.state["failed"]:
            self.state["unprocessed"].append(topic)
            self.save_state()

    def add_topics(self, topics: List[str]) -> None:
        """
        Добавляет список новых тем.
        """
        for topic in topics:
            self.add_topic(topic)

    def get_next_unprocessed_topic(self) -> Optional[str]:
        """
        Возвращает следующую необработанную тему, либо None.
        """
        return self.state["unprocessed"][0] if self.state["unprocessed"] else None

    def get_unprocessed_topics(self) -> List[str]:
        """
        Возвращает список необработанных тем.
        """
        return list(self.state["unprocessed"])

    def get_processed_topics(self) -> List[str]:
        """
        Возвращает список успешно обработанных тем.
        """
        return list(self.state["processed"].keys())

    def get_failed_topics(self) -> List[str]:
        """
        Возвращает список тем с ошибками.
        """
        return list(self.state["failed"].keys())

    def reset_failed_topics(self) -> None:
        """
        Переносит все failed темы обратно в unprocessed, очищает failed.
        """
        failed = self.get_failed_topics()
        # Не добавлять дубликаты
        for topic in failed:
            if topic not in self.state["unprocessed"]:
                self.state["unprocessed"].append(topic)
        self.state["failed"] = {}
        self.save_state()
        self.logger.info("Failed topics reset to unprocessed.")

    # ----------- Группа: Методы для статистики и статуса -----------
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику обработки тем.
        """
        return dict(self.state["statistics"])

    def add_processing_stats(self, stats: dict) -> None:
        """
        Обновляет (дополняет) статистику.
        """
        self.state["statistics"].update(stats)
        self.save_state()

    def get_system_uptime(self) -> float:
        """
        Возвращает аптайм системы (в секундах) с момента старта.
        """
        try:
            start = datetime.fromisoformat(self.state["statistics"]["start_time"])
        except Exception:
            start = datetime.utcnow()
        return (datetime.utcnow() - start).total_seconds()

    def set_system_status(self, status: str) -> None:
        """
        Устанавливает строковый статус системы ("RUNNING", "SHUTDOWN", "ERROR" и т.д.).
        """
        self.state["system_status"] = status
        self.state["last_activity"] = datetime.utcnow().isoformat()
        self.save_state()

    def get_system_status(self) -> str:
        """
        Возвращает текущий статус системы.
        """
        return self.state.get("system_status", "UNKNOWN")

    def get_last_activity(self) -> datetime:
        """
        Возвращает timestamp последней активности.
        """
        try:
            return datetime.fromisoformat(self.state["last_activity"])
        except Exception:
            return datetime.utcnow()

    # ----------- Группа: Бэкапы, восстановление и диагностика -----------
    def backup_state(self) -> str:
        """
        Делаем резервную копию состояния (сохраняет как <state_file>.backup.json)
        """
        backup_path = self.state_file.with_suffix(".backup.json")
        try:
            with backup_path.open("w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=4, ensure_ascii=False)
            self.logger.info(f"State backup saved to {backup_path}")
            return str(backup_path)
        except Exception as e:
            self.logger.error("Failed to backup state.", exc_info=True)
            return ""

    def restore_state(self, backup_path: Union[str, Path]) -> bool:
        """
        Восстанавливает состояние из резервной копии.
        """
        path = Path(backup_path)
        if not path.exists():
            self.logger.error(f"Backup file not found: {backup_path}")
            return False

        try:
            with path.open("r", encoding="utf-8") as f:
                self.state = json.load(f)
            self._ensure_state_integrity()
            self.save_state()
            self.logger.info("State restored from backup.")
            return True
        except Exception as e:
            self.logger.error("Failed to restore state.", exc_info=True)
            return False

    # ----------- Группа: Вспомогательные методы доступа (get/set/delete) -----------
    def get(self, key: str, default: Any = None) -> Any:
        """
        Получить значение по ключу из состояния (верхний уровень).
        """
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Установить значение по ключу в состоянии (верхний уровень).
        """
        self.state[key] = value
        self.save_state()

    def delete(self, key: str) -> None:
        """
        Удалить ключ из состояния (верхний уровень).
        """
        if key in self.state:
            del self.state[key]
            self.save_state()

    # ----------- Группа: Диагностика и вывод состояния -----------
    def dump_state(self) -> str:
        """
        Возвращает строковое представление состояния (удобно для отладки).
        """
        return json.dumps(self.state, indent=2, ensure_ascii=False)

    def print_state(self) -> None:
        """
        Печатает состояние в лог (информационный уровень).
        """
        self.logger.info(f"STATE DUMP:\n{self.dump_state()}")
