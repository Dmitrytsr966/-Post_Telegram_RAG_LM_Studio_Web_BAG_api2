from .base import BaseHook

class RemoveStopwordsHook(BaseHook):
    """
    Удаляет стоп-слова из текста.
    params: stopwords - множество стоп-слов (по умолчанию - английские).
    """
    params = {}
    conflicts = set()

    def __init__(self, stopwords: set = None):
        if stopwords is None:
            # Простейший набор стандартных английских стоп-слов
            self.stopwords = {
                "the", "a", "an", "and", "is", "in", "on", "for", "at", "of", "to", "with", "by", "from"
            }
        else:
            self.stopwords = set(stopwords)
        self.params["stopwords"] = self.stopwords

    def __call__(self, text: str, meta: dict, **context) -> str:
        tokens = text.split()
        filtered = [w for w in tokens if w.lower() not in self.stopwords]
        return " ".join(filtered)

    def summary(self, old_text: str, new_text: str) -> str:
        return "Stopwords removed"
