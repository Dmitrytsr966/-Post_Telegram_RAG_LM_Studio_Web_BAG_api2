from .base import BaseHook
import logging
import re
from typing import Optional, Dict, Any, List

class RemoveExtraSpacesHook(BaseHook):
    """
    Хук для удаления лишних пробелов и невидимых символов в тексте.
    - Удаляет повторяющиеся пробелы, табуляции, неразрывные пробелы, нулевые ширины, лишние отступы до/после строк.
    - Конфигурируемые символы для очистки.
    - Логирует количество изменений.
    """

    params = {
        "remove_nbsp": True,         # Удалять &nbsp; и \xa0
        "remove_zero_width": True,   # Удалять невидимые символы Unicode
        "collapse_spaces": True,     # Схлопывать повторяющиеся пробелы и табы
        "log_level": logging.INFO
    }
    conflicts = set()

    # Unicode invisible characters (zero-width space, joiners, etc.)
    ZERO_WIDTH_RE = r'[\u200B-\u200D\uFEFF]'

    def __init__(
        self,
        remove_nbsp: bool = True,
        remove_zero_width: bool = True,
        collapse_spaces: bool = True,
        log_level: int = logging.INFO,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.remove_nbsp = remove_nbsp
        self.remove_zero_width = remove_zero_width
        self.collapse_spaces = collapse_spaces
        self.log_level = log_level
        self.logger = logging.getLogger(self.__class__.__name__)

    def _clean_line(self, line: str) -> str:
        original = line
        # Remove non-breaking spaces (&nbsp;, \xa0)
        if self.remove_nbsp:
            line = line.replace('\xa0', ' ').replace('&nbsp;', ' ')
        # Remove zero-width characters
        if self.remove_zero_width:
            line = re.sub(self.ZERO_WIDTH_RE, '', line)
        # Collapse multiple spaces and tabs
        if self.collapse_spaces:
            line = re.sub(r'[ \t]+', ' ', line)
        # Remove trailing/leading spaces
        line = line.strip()
        return line

    def __call__(self, text: str, meta: Optional[Dict[str, Any]] = None, **context) -> str:
        """
        :param text: исходный текст
        :param meta: метаданные (дополняются количеством изменений)
        :return: очищенный текст
        """
        lines = text.splitlines()
        cleaned_lines = []
        changed_count = 0
        for line in lines:
            cleaned = self._clean_line(line)
            if cleaned != line:
                changed_count += 1
            cleaned_lines.append(cleaned)
        # Remove multiple empty lines in a row
        out = '\n'.join(cleaned_lines)
        out = re.sub(r'\n{2,}', '\n', out)
        if meta is not None:
            meta.setdefault("hook_stats", {})[self.__class__.__name__] = {
                "changed_lines": changed_count,
                "total_lines": len(lines)
            }
        self.logger.log(self.log_level, f"RemoveExtraSpacesHook: changed {changed_count} of {len(lines)} lines")
        return out.strip()

    def summary(self, old_text: str, new_text: str) -> str:
        old_len = len(old_text)
        new_len = len(new_text)
        diff = old_len - new_len
        return f"RemoveExtraSpacesHook: reduced size by {diff} characters ({old_len} → {new_len})"
