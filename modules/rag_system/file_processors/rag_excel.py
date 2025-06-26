import logging
import pandas as pd
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

class ExcelFileProcessor:
    def __init__(self, config: dict, validator=None, hooks: Optional[List] = None):
        self.config = config
        self.rag_cfg = config.get("rag", {})
        self.validator_cfg = config.get("content_validator", {})
        self.chunk_size = self.rag_cfg.get("chunk_size", 512)
        self.chunk_overlap = self.rag_cfg.get("chunk_overlap", 50)
        self.max_context_length = self.rag_cfg.get("max_context_length", 4096)
        self.media_context_length = self.rag_cfg.get("media_context_length", 1024)
        self.similarity_threshold = self.rag_cfg.get("similarity_threshold", 0.7)
        self.remove_tables = self.validator_cfg.get("remove_tables", True)
        self.max_length_no_media = self.validator_cfg.get("max_length_no_media", 4096)
        self.max_length_with_media = self.validator_cfg.get("max_length_with_media", 1024)
        self.validator = validator
        self.hooks = hooks or []
        self.logger = logging.getLogger("rag_excel")

    def extract_text(self, file_path: str, **kwargs) -> dict:
        result = {
            "text": "",
            "chunks": [],
            "success": False,
            "error": None,
            "cleaned": False,
            "meta": {
                "file_path": str(file_path),
                "file_type": "excel",
                "parser": "rag_excel",
                "lines": 0,
                "chars": 0,
                "percent_empty": 0.0,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "max_context_length": self.max_context_length,
                "media_context_length": self.media_context_length,
                "removed_tables": False,
                "hooks_applied": [],
            }
        }
        try:
            df = pd.read_excel(file_path, engine="openpyxl")
            orig_rows = len(df)
            df = df.dropna(how='all')
            def is_useless_row(row):
                return all((str(cell).strip() == "" or str(cell).strip() == "nan") for cell in row)
            cleaned_df = df[~df.apply(is_useless_row, axis=1)]
            percent_empty = 100.0 * (orig_rows - len(cleaned_df)) / (orig_rows or 1)
            cleaned_df = self.clean_dataframe(cleaned_df)
            text = cleaned_df.to_string(index=False)
            # Remove tables if configured
            if self.remove_tables and self.validator:
                text, removed = self.validator.remove_tables(text)
                result["meta"]["removed_tables"] = removed
            # Применяем хуки, если есть
            for hook in self.hooks:
                text = hook(text)
                result["meta"]["hooks_applied"].append(getattr(hook, "__class__", type(hook)).__name__)
            result["text"] = text.strip()
            result["success"] = True
            result["cleaned"] = (len(cleaned_df) != orig_rows)
            result["meta"]["lines"] = len(cleaned_df)
            result["meta"]["chars"] = len(result["text"])
            result["meta"]["percent_empty"] = percent_empty
            # Чанкинг, если надо
            result["chunks"] = self.chunk_text(result["text"])
        except Exception as e:
            result["error"] = str(e)
            self.logger.warning(f"EXCEL extraction failed for {file_path}: {e}")
        return result

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        def clean_cell(cell):
            cleaned = re.sub(r'<[^>]+>', '', str(cell)).strip()
            return cleaned
        # Современная замена applymap:
        return df.apply(lambda col: col.map(clean_cell))

    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        size = self.chunk_size
        overlap = self.chunk_overlap
        tokens = text.split('\n')
        i = 0
        while i < len(tokens):
            chunk = '\n'.join(tokens[i : i + size])
            chunks.append(chunk)
            i += size - overlap
        return [c for c in chunks if c.strip()]

    def validate_content(self, text: str, has_media: bool = False) -> bool:
        limit = self.max_length_with_media if has_media else self.max_length_no_media
        if self.validator:
            return self.validator.validate_length(text, has_media)
        return len(text) <= limit

    def save_debug_info(self, meta: dict):
        self.logger.debug(f"Meta info: {meta}")

# Для совместимости со старой архитектурой
_default_processor = None

def extract_text(file_path: str, config: dict = None, validator=None, hooks=None, **kwargs) -> dict:
    global _default_processor
    if _default_processor is None or (_default_processor.config != config):
        _default_processor = ExcelFileProcessor(config or {}, validator=validator, hooks=hooks)
    return _default_processor.extract_text(file_path)
