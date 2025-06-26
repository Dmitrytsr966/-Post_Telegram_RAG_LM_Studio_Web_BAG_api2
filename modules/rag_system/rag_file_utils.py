import logging
from pathlib import Path
from glob import glob
from typing import List, Dict, Any, Optional

from .file_processor_manager import FileProcessorManager
from .hook_manager import HookManager
from .hooks import ALL_HOOKS

# Импорт парсеров по расширениям
from .file_processors.rag_txt import extract_text as txt_parser
from .file_processors.rag_csv import extract_text as csv_parser
from .file_processors.rag_excel import extract_text as excel_parser
from .file_processors.rag_pdf import extract_text as pdf_parser
from .file_processors.rag_docx import extract_text as docx_parser
from .file_processors.rag_html import extract_text as html_parser
from .file_processors.rag_markdown import extract_text as markdown_parser
from .file_processors.rag_json import extract_text as json_parser
from .file_processors.rag_pptx import extract_text as pptx_parser
from .file_processors.rag_audio import extract_text as audio_parser
from .file_processors.rag_video import extract_text as video_parser
from .file_processors.rag_fallback_textract import extract_text as textract_fallback

class RAGFileUtils:
    """
    Мощная обёртка для обработки файлов с поддержкой хуков, batch-режимов и сбора статистики.
    Позволяет легко масштабировать парсинг разных форматов и расширять pipeline обработки.
    """
    def __init__(self, logger: Optional[logging.Logger] = None, hooks_config: Optional[dict] = None):
        self.logger = logger or logging.getLogger("RAGFileUtils")
        self.hook_manager = HookManager(self.logger)
        self.manager = FileProcessorManager(self.logger, self.hook_manager)

        # Регистрация парсеров по расширениям
        self._register_all_parsers()

        # Регистрация хуков из конфигурации, если есть
        if hooks_config:
            self._register_hooks_from_config(hooks_config)

    def _register_all_parsers(self):
        self.manager.register_parser([".txt"], txt_parser)
        self.manager.register_parser([".csv"], csv_parser)
        self.manager.register_parser([".xlsx", ".xls"], excel_parser)
        self.manager.register_parser([".pdf"], pdf_parser)
        self.manager.register_parser([".docx"], docx_parser)
        self.manager.register_parser([".html", ".htm"], html_parser)
        self.manager.register_parser([".md", ".markdown"], markdown_parser)
        self.manager.register_parser([".json"], json_parser)
        self.manager.register_parser([".pptx"], pptx_parser)
        self.manager.register_parser([".mp3", ".wav", ".flac"], audio_parser)
        self.manager.register_parser([".mp4", ".avi", ".mov"], video_parser)
        self.manager.register_fallback(textract_fallback)

    def _register_hooks_from_config(self, hooks_config: dict):
        for hook_name, params in hooks_config.get("pre", {}).items():
            hook_cls = ALL_HOOKS[hook_name]
            self.hook_manager.register_pre_hook(hook_cls(**params))
        for hook_name, params in hooks_config.get("post", {}).items():
            hook_cls = ALL_HOOKS[hook_name]
            self.hook_manager.register_post_hook(hook_cls(**params))

    def extract_text(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Извлекает текст и метаданные из любого поддерживаемого файла.
        Возвращает dict с ключами text, success, error, meta.
        """
        return self.manager.extract_text(file_path, **kwargs)

    def extract_text_batch(
        self,
        dir_path: str,
        pattern: str = "**/*",
        recursive: bool = True,
        skip_partial: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Batch-обработка всех файлов в директории (или по паттерну).
        Возвращает список dict-результатов.
        """
        search_path = str(Path(dir_path) / pattern)
        files = glob(search_path, recursive=recursive)
        if not files:
            self.logger.warning(f"No files found for pattern {pattern} in {dir_path}")
        results = self.manager.extract_text_batch(files, skip_partial=skip_partial, **kwargs)
        return results

    def get_supported_extensions(self) -> List[str]:
        """Возвращает список всех поддерживаемых расширений файлов."""
        return self.manager.get_supported_extensions()

    def get_stats_from_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Агрегирует статистику по batch-результатам: success_rate, avg_length, lang_distribution и др.
        """
        total = len(results)
        success = sum(1 for r in results if r.get("success"))
        partial = sum(1 for r in results if r.get("meta", {}).get("partial_success"))
        errors = [r.get("error") for r in results if not r.get("success")]
        lengths = [r.get("meta", {}).get("chars", 0) for r in results if r.get("success")]
        langs = {}
        for r in results:
            lang = r.get("meta", {}).get("lang")
            if lang:
                langs[lang] = langs.get(lang, 0) + 1
        stats = {
            "total": total,
            "success": success,
            "partial": partial,
            "success_rate": round(success/total, 3) if total else 0,
            "avg_length": round(sum(lengths)/success, 1) if success else 0,
            "lang_distribution": langs,
            "errors": errors,
        }
        return stats

    def get_session_errors(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Возвращает список ошибок по batch-результатам.
        """
        return [
            {
                "file": r.get("meta", {}).get("file_path"),
                "reason": r.get("error"),
                "partial": r.get("meta", {}).get("partial_success", False)
            }
            for r in results if not r.get("success")
        ]

    def filter_results_by_lang(self, results: List[Dict[str, Any]], lang_code: str) -> List[Dict[str, Any]]:
        """
        Фильтрует batch-результаты по языку (по коду lang).
        """
        return [r for r in results if r.get("meta", {}).get("lang") == lang_code]

    def filter_results_by_success(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Возвращает только успешные результаты из списка batch-результатов.
        """
        return [r for r in results if r.get("success")]

    def save_results_to_json(self, results: List[Dict[str, Any]], out_path: str) -> None:
        """
        Сохраняет batch-результаты в JSON-файл.
        """
        import json
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Batch results saved to {out_path}")
        except Exception as e:
            self.logger.error(f"Failed to save batch results: {e}")

class FileProcessor:
    """
    Адаптер между RAGRetriever и RAGFileUtils.
    Позволяет вызывать функционал извлечения текста из файла и валидации поддерживаемых расширений.
    """
    def __init__(self):
        self.utils = RAGFileUtils()

    def extract_text_from_file(self, file_path: str) -> str:
        result = self.utils.extract_text(file_path)
        return result.get("text", "")

    def validate_file(self, file_path: str) -> bool:
        supported = self.utils.get_supported_extensions()
        return any(file_path.lower().endswith(ext) for ext in supported)
