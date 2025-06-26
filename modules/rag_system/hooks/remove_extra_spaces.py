from .base import BaseHook
import re

class RemoveExtraSpacesHook(BaseHook):
    """
    Удаляет лишние пробелы, табуляции, и повторяющиеся пробелы между словами.
    """
    params = {}
    conflicts = set()

    def __call__(self, text: str, meta: dict, **context) -> str:
        # Удаляет повторяющиеся пробелы/табы, нормализует отступы
        out = re.sub(r'[ \t]+', ' ', text)
        out = re.sub(r' *\n *', '\n', out)  # пробелы до/после переноса строки
        # Удаляем несколько пустых строк подряд
        out = re.sub(r'\n{2,}', '\n', out)
        return out.strip()

    def summary(self, old_text: str, new_text: str) -> str:
        return "Extra spaces removed"
