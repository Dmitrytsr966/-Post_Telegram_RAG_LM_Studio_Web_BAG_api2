import logging
import pandas as pd
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple, Union

# --- Custom Exceptions ---

class ParserError(Exception):
    """Base exception for parsing errors."""
    pass

class FileReadError(ParserError):
    pass

class DataCleaningError(ParserError):
    pass

class HookApplicationError(ParserError):
    pass

class ChunkingError(ParserError):
    pass

# --- Logging Mixin ---

class LoggingMixin:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def log(self, message, level=logging.INFO):
        if self.logger:
            self.logger.log(level, message)

# --- Excel Reader Abstraction ---

class IExcelFileReader(LoggingMixin):
    def read(self, file_path: str) -> pd.DataFrame:
        raise NotImplementedError

class PandasExcelFileReader(IExcelFileReader):
    def read(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_excel(file_path, engine="openpyxl")
            self.log(f"Read Excel file '{file_path}', shape: {df.shape}")
            return df
        except Exception as e:
            self.log(f"Failed to read Excel file '{file_path}': {e}", level=logging.ERROR)
            raise FileReadError(f"Failed to read Excel file: {e}")

# --- Data Cleaner ---

class DataCleaner(LoggingMixin):
    DEFAULT_MIN_CELL_LENGTH = 1

    def __init__(self, min_cell_length: int = None, remove_html: bool = True, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.min_cell_length = min_cell_length or self.DEFAULT_MIN_CELL_LENGTH
        self.remove_html = remove_html

    def is_useless_row(self, row: pd.Series) -> bool:
        # True if all cells are empty, whitespace, or nan
        return all(self.is_useless_cell(cell) for cell in row)

    def is_useless_cell(self, cell: Any) -> bool:
        s = str(cell).strip()
        if not s or s.lower() == 'nan':
            return True
        # Only html tags or only whitespace
        if re.fullmatch(r'<[^>]+>', s):
            return True
        if not re.search(r'\w', s):  # no letters/numbers
            return True
        if len(s) < self.min_cell_length:
            return True
        return False

    def clean_cell(self, cell: Any) -> str:
        s = str(cell)
        if self.remove_html:
            s = re.sub(r'<[^>]+>', '', s)
            s = re.sub(r'&[a-zA-Z0-9#]+;', '', s)
        return s.strip()

    def clean_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
        orig_row_count = len(df)
        # Remove all-na rows
        df = df.dropna(how="all")
        # Remove rows where all cells are useless
        mask = ~df.apply(self.is_useless_row, axis=1)
        filtered = df[mask]
        removed_rows = orig_row_count - len(filtered)
        self.log(f"Removed {removed_rows} useless rows out of {orig_row_count}")
        # Clean cells
        cleaned = filtered.applymap(self.clean_cell)
        # Remove columns that are all empty after cleaning
        cleaned = cleaned.dropna(axis=1, how='all')
        removed_cols = filtered.shape[1] - cleaned.shape[1]
        if removed_cols > 0:
            self.log(f"Removed {removed_cols} empty columns after cleaning")
        return cleaned, removed_rows, removed_cols

# --- Hook Applier ---

class HookApplier(LoggingMixin):
    def __init__(self, hooks: Optional[List[Callable]] = None, logger: Optional[logging.Logger]=None):
        super().__init__(logger)
        self.hooks = hooks or []

    def apply(self, text: str, meta: dict) -> Tuple[str, List[str]]:
        applied = []
        for hook in self.hooks:
            try:
                new_text = hook(text, meta)
                applied.append(getattr(hook, "__class__", type(hook)).__name__)
                text = new_text
            except Exception as e:
                self.log(f"Error applying hook {hook}: {e}", level=logging.WARNING)
                raise HookApplicationError(f"Hook '{hook}' failed: {e}")
        return text, applied

# --- Chunker ---

class Chunker(LoggingMixin):
    def __init__(self, chunk_size: int, chunk_overlap: int, logger: Optional[logging.Logger]=None):
        super().__init__(logger)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        lines = text.split('\n')
        chunks = []
        i = 0
        n = len(lines)
        while i < n:
            chunk = '\n'.join(lines[i: i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            i += max(1, self.chunk_size - self.chunk_overlap)
        self.log(f"Chunked text into {len(chunks)} chunks (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks

# --- Main Processor ---

class ExcelFileProcessor(LoggingMixin):
    def __init__(
        self,
        config: dict,
        validator=None,
        hooks: Optional[List[Callable]] = None,
        logger: Optional[logging.Logger] = None,
        excel_reader: Optional[IExcelFileReader] = None,
        data_cleaner: Optional[DataCleaner] = None,
        hook_applier: Optional[HookApplier] = None,
        chunker: Optional[Chunker] = None
    ):
        super().__init__(logger)
        self.config = config
        self.rag_cfg = config.get("rag", {})
        self.validator_cfg = config.get("content_validator", {})
        # SOLID: параметры только через конструктор
        self.chunk_size = self.rag_cfg.get("chunk_size", 512)
        self.chunk_overlap = self.rag_cfg.get("chunk_overlap", 50)
        self.max_context_length = self.rag_cfg.get("max_context_length", 4096)
        self.media_context_length = self.rag_cfg.get("media_context_length", 1024)
        self.similarity_threshold = self.rag_cfg.get("similarity_threshold", 0.7)
        self.remove_tables = self.validator_cfg.get("remove_tables", True)
        self.max_length_no_media = self.validator_cfg.get("max_length_no_media", 4096)
        self.max_length_with_media = self.validator_cfg.get("max_length_with_media", 1024)
        self.validator = validator

        # Абстракции
        self.excel_reader = excel_reader or PandasExcelFileReader(self.logger)
        self.data_cleaner = data_cleaner or DataCleaner(logger=self.logger)
        self.hook_applier = hook_applier or HookApplier(hooks, logger=self.logger)
        self.chunker = chunker or Chunker(self.chunk_size, self.chunk_overlap, logger=self.logger)

    def extract_text(self, file_path: str, **kwargs) -> dict:
        """
        Основная функция: читает Excel-файл, очищает строки, применяет хуки, чанкует.
        Возвращает dict:
            {
                "text": str,
                "chunks": List[str],
                "success": bool,
                "error": Optional[str],
                "cleaned": bool,
                "meta": dict,
            }
        """
        meta = {
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
        try:
            # 1. Read file
            df = self.excel_reader.read(file_path)
            orig_rows = len(df)
            if orig_rows == 0:
                raise ParserError("Excel file is empty.")

            # 2. Clean data
            cleaned_df, removed_rows, removed_cols = self.data_cleaner.clean_dataframe(df)
            percent_empty = 100.0 * (removed_rows / (orig_rows or 1))
            meta["lines"] = len(cleaned_df)
            meta["percent_empty"] = percent_empty
            meta["removed_rows"] = removed_rows
            meta["removed_cols"] = removed_cols
            if len(cleaned_df) == 0:
                raise DataCleaningError("All rows were removed as empty or useless.")

            # 3. Convert to text representation
            text = cleaned_df.to_string(index=False)
            meta["chars"] = len(text)

            # 4. Remove tables if configured
            if self.remove_tables and self.validator and hasattr(self.validator, "remove_tables"):
                text, removed = self.validator.remove_tables(text)
                meta["removed_tables"] = removed

            # 5. Apply hooks
            text, hooks_applied = self.hook_applier.apply(text, meta)
            meta["hooks_applied"] = hooks_applied

            # 6. Chunking
            chunks = self.chunker.chunk_text(text)

            # 7. Validate content length
            cleaned = (removed_rows > 0 or removed_cols > 0 or len(hooks_applied) > 0)
            result = {
                "text": text.strip(),
                "chunks": chunks,
                "success": True,
                "error": None,
                "cleaned": cleaned,
                "meta": meta,
            }
            return result

        except ParserError as e:
            self.log(f"Parser error: {e}", level=logging.WARNING)
            return {
                "text": "",
                "chunks": [],
                "success": False,
                "error": str(e),
                "cleaned": False,
                "meta": meta,
            }
        except Exception as e:
            self.log(f"Unexpected error in ExcelFileProcessor: {e}", level=logging.ERROR)
            return {
                "text": "",
                "chunks": [],
                "success": False,
                "error": f"Unexpected error: {e}",
                "cleaned": False,
                "meta": meta,
            }

    def validate_content(self, text: str, has_media: bool = False) -> bool:
        """
        Проверяет, не превышает ли текст максимальную длину (используется для фильтрации чанков).
        """
        limit = self.max_length_with_media if has_media else self.max_length_no_media
        if self.validator and hasattr(self.validator, "validate_length"):
            return self.validator.validate_length(text, has_media)
        return len(text) <= limit

# --- Adapter for backward compatibility ---

_default_processor = None

def extract_text(file_path: str, config: dict = None, validator=None, hooks=None, **kwargs) -> dict:
    global _default_processor
    config = config or {}
    logger = logging.getLogger("rag_excel")
    if _default_processor is None or (_default_processor.config != config):
        _default_processor = ExcelFileProcessor(config, validator=validator, hooks=hooks, logger=logger)
    return _default_processor.extract_text(file_path)
