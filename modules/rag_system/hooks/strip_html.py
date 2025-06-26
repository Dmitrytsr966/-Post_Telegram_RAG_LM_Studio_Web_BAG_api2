from .base import BaseHook
import re

class StripHTMLHook(BaseHook):
    """
    Удаляет HTML-теги из текста.
    """
    params = {}
    conflicts = {"StripMarkdownHook"}

    HTML_TAG_RE = re.compile(r'<[^>]+>')

    def __call__(self, text: str, meta: dict, **context) -> str:
        return self.HTML_TAG_RE.sub("", text)

    def summary(self, old_text: str, new_text: str) -> str:
        return "HTML tags stripped"
