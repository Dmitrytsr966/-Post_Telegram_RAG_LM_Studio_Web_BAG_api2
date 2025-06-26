from .base import BaseHook
import string

class RemovePunctuationHook(BaseHook):
    """
    Удаляет всю пунктуацию из текста.
    """
    params = {}
    conflicts = set()

    def __call__(self, text: str, meta: dict, **context) -> str:
        table = str.maketrans("", "", string.punctuation)
        return text.translate(table)

    def summary(self, old_text: str, new_text: str) -> str:
        return "Punctuation removed"
