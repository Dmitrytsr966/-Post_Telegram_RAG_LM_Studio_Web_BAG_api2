from .base import BaseHook

class RemoveEmptyLinesHook(BaseHook):
    """
    Удаляет пустые строки из текста.
    """
    params = {}
    conflicts = set()

    def __call__(self, text: str, meta: dict, **context) -> str:
        lines = text.splitlines()
        cleaned = [line for line in lines if line.strip()]
        return "\n".join(cleaned)

    def summary(self, old_text: str, new_text: str) -> str:
        return "Empty lines removed"
