from .base import BaseHook
import re

class CustomReplaceHook(BaseHook):
    """
    Производит пользовательские замены по строке или regex.
    params: replacements - список кортежей или словарей вида (pattern, repl[, is_regex])
    """
    params = {}
    conflicts = set()

    def __init__(self, replacements=None):
        """
        replacements: list of (pattern, repl, is_regex) or (pattern, repl)
        """
        if replacements is None:
            self.replacements = []
        else:
            self.replacements = list(replacements)
        self.params["replacements"] = self.replacements

    def __call__(self, text: str, meta: dict, **context) -> str:
        out = text
        for item in self.replacements:
            if len(item) == 3:
                pattern, repl, is_regex = item
            else:
                pattern, repl = item
                is_regex = False
            if is_regex:
                out = re.sub(pattern, repl, out)
            else:
                out = out.replace(pattern, repl)
        return out

    def summary(self, old_text: str, new_text: str) -> str:
        return f"Custom replacements ({len(self.replacements)}) applied"
