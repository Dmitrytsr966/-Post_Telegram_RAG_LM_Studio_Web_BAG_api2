from .base import BaseHook
import logging
import re
from typing import Optional, Dict, Any, List

class RemoveEmptyLinesHook(BaseHook):
    """
    Хук для удаления пустых и "мусорных" строк из текста.
    - Удаляет строки, содержащие только пробелы, табуляции, спецсимволы или ничего.
    - Логирует количество удалённых строк.
    - Поддерживает конфигурируемые паттерны для расширенного определения "пустых" строк.
    """

    # Параметры по умолчанию
    params = {
        "extra_patterns": None,   # список regexp для дополнительных фильтров мусора
        "log_level": logging.INFO
    }
    conflicts = set()

    def __init__(self, extra_patterns: Optional[List[str]] = None, log_level: int = logging.INFO, **kwargs):
        """
        :param extra_patterns: список regexp-паттернов для определения "мусорных" строк (например, только спецсимволы)
        :param log_level: уровень логирования для отчёта об удалённых строках
        """
        super().__init__(**kwargs)
        self.extra_patterns = extra_patterns or []
        self.log_level = log_level
        self.logger = logging.getLogger(self.__class__.__name__)

    def _is_empty_or_junk(self, line: str) -> bool:
        """Проверяет, является ли строка пустой, из пробелов, табов, или подходит под дополнительные паттерны."""
        s = line.strip()
        if not s:
            return True
        # По умолчанию: только спецсимволы (например, ---- или ===)
        if re.fullmatch(r'[-=*_~\s]+', s):
            return True
        # Дополнительные мусорные паттерны
        for pat in self.extra_patterns:
            if re.fullmatch(pat, s):
                return True
        return False

    def __call__(self, text: str, meta: Optional[Dict[str, Any]] = None, **context) -> str:
        """
        :param text: исходный текст
        :param meta: метаданные (дополняются количеством удалённых строк)
        :return: очищенный текст
        """
        lines = text.splitlines()
        cleaned_lines = []
        removed = 0
        for line in lines:
            if not self._is_empty_or_junk(line):
                cleaned_lines.append(line)
            else:
                removed += 1
        if meta is not None:
            meta.setdefault("hook_stats", {})[self.__class__.__name__] = {
                "removed_lines": removed,
                "total_lines": len(lines)
            }
        self.logger.log(self.log_level, f"RemoveEmptyLinesHook: removed {removed} of {len(lines)} lines")
        return "\n".join(cleaned_lines)

    def summary(self, old_text: str, new_text: str) -> str:
        old_count = len(old_text.splitlines())
        new_count = len(new_text.splitlines())
        removed = old_count - new_count
        return f"RemoveEmptyLinesHook: removed {removed} empty/junk lines ({old_count} → {new_count})"
