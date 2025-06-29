import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

class MarkdownCleaner:
    """
    Отвечает за удаление markdown/HTML и "мусорных" конструкций из текста.
    Конфигурируемый набор фильтров для гибкости.
    """

    def __init__(
        self,
        remove_html: bool = True,
        remove_markdown: bool = True,
        remove_code_blocks: bool = True,
        remove_tables: bool = True,
        min_line_length: int = 1,
        junk_patterns: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.remove_html = remove_html
        self.remove_markdown = remove_markdown
        self.remove_code_blocks = remove_code_blocks
        self.remove_tables = remove_tables
        self.min_line_length = min_line_length
        self.junk_patterns = junk_patterns or []
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def clean(self, text: str) -> str:
        orig_len = len(text)
        # 1. Remove code blocks (```...```)
        if self.remove_code_blocks:
            text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        # 2. Remove markdown tables (| ... |), including multi-line
        if self.remove_tables:
            text = re.sub(r'((\|.*\|.*\n?)+)', '', text)
        # 3. Remove HTML tags
        if self.remove_html:
            text = re.sub(r'<[^>]+>', '', text)
        # 4. Remove markdown headings, blockquotes, lists, decorations
        if self.remove_markdown:
            patterns = [
                r'^\s*#+\s+',         # headings
                r'^\s*>\s+',          # blockquotes
                r'^\s*[*\-+]\s+',     # unordered lists
                r'^\s*\d+\.\s+',      # ordered lists
                r'`[^`]*`',           # inline code
                r'[*_]{2,}',          # bold/italic
                r'!\[[^\]]*\]\([^\)]*\)', # images
                r'\[[^\]]*\]\([^\)]*\)',  # links
                r'^---+$',            # horizontal rules
                r'^___+$',
                r'^\*\*\*+$'
            ]
            for pat in patterns:
                text = re.sub(pat, '', text, flags=re.MULTILINE)
        # 5. Remove user-defined junk patterns
        for pat in self.junk_patterns:
            text = re.sub(pat, '', text, flags=re.MULTILINE)
        # 6. Clean up lines: remove empty, too short, or only special chars
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            s = line.strip()
            if not s or len(s) < self.min_line_length:
                continue
            # Remove lines that are only special characters
            if re.fullmatch(r'[-=*_~\s]+', s):
                continue
            cleaned_lines.append(s)
        self.logger.info(f"MarkdownCleaner: reduced text from {orig_len} to {len('\n'.join(cleaned_lines))} chars")
        return "\n".join(cleaned_lines)

class MarkdownFileProcessor:
    """
    Главный класс для обработки markdown-файлов:
    - Читает, очищает, возвращает текст и метаданные.
    """

    def __init__(self, cleaner: Optional[MarkdownCleaner] = None, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.cleaner = cleaner or MarkdownCleaner(logger=self.logger)

    def extract_text(self, file_path: str, **kwargs) -> Dict[str, Any]:
        meta = {
            "file_path": file_path,
            "file_type": "md",
            "parser": "rag_markdown",
            "lines": 0,
            "chars": 0,
            "cleaned": False
        }
        try:
            p = Path(file_path)
            text = p.read_text(encoding=kwargs.get("encoding", "utf-8"))
            cleaned_text = self.cleaner.clean(text)
            meta["lines"] = cleaned_text.count('\n') + 1 if cleaned_text else 0
            meta["chars"] = len(cleaned_text)
            meta["cleaned"] = cleaned_text != text
            self.logger.info(
                f"Processed markdown file: {file_path}, lines: {meta['lines']}, chars: {meta['chars']}, cleaned: {meta['cleaned']}"
            )
            return {
                "text": cleaned_text,
                "success": True,
                "error": None,
                "cleaned": meta["cleaned"],
                "meta": meta
            }
        except Exception as e:
            self.logger.warning(f"Markdown extraction failed for {file_path}: {e}")
            meta["error"] = str(e)
            return {
                "text": "",
                "success": False,
                "error": str(e),
                "cleaned": False,
                "meta": meta
            }

# Adapter for pipeline compatibility
_default_processor = None

def extract_text(file_path: str, cleaner: MarkdownCleaner = None, **kwargs) -> Dict[str, Any]:
    global _default_processor
    if _default_processor is None:
        _default_processor = MarkdownFileProcessor(cleaner=cleaner)
    return _default_processor.extract_text(file_path, **kwargs)
