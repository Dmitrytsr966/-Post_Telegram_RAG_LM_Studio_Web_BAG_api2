from .base import BaseHook

class LowerCaseHook(BaseHook):
    """
    Приводит текст к нижнему регистру.
    """
    params = {}
    conflicts = set()

    def __call__(self, text: str, meta: dict, **context) -> str:
        return text.lower()

    def summary(self, old_text: str, new_text: str) -> str:
        return "Text converted to lower case"
