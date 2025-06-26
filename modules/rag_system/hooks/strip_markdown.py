from .base import BaseHook
import re

class StripMarkdownHook(BaseHook):
    """
    Удаляет основные элементы Markdown из текста.
    """
    params = {}
    conflicts = {"StripHTMLHook"}

    # Базовые markdown паттерны для удаления
    MARKDOWN_RE_LIST = [
        (re.compile(r'`{1,3}.*?`{1,3}', re.DOTALL), ""),        # код в `` или ```
        (re.compile(r'!\[[^\]]*\]\([^\)]*\)'), ""),             # картинки ![alt](url)
        (re.compile(r'\[[^\]]*\]\([^\)]*\)'), ""),              # ссылки [text](url)
        (re.compile(r'#+ '), ""),                               # заголовки
        (re.compile(r'\*{1,2}([^*]+)\*{1,2}'), r'\1'),          # *выделение* и **выделение**
        (re.compile(r'_{1,2}([^_]+)_{1,2}'), r'\1'),            # _выделение_ и __выделение__
        (re.compile(r'> .+'), ""),                              # цитаты
        (re.compile(r'^[-*] ', re.MULTILINE), ""),              # списки
    ]

    def __call__(self, text: str, meta: dict, **context) -> str:
        out = text
        for rex, repl in self.MARKDOWN_RE_LIST:
            out = rex.sub(repl, out)
        return out

    def summary(self, old_text: str, new_text: str) -> str:
        return "Markdown formatting stripped"
