import logging
from pathlib import Path
from glob import glob
from typing import List, Dict, Any, Optional, Callable, Tuple, Union

from .file_processor_manager import FileProcessorManager
from .hook_manager import HookManager, HookResult
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
    Позволяет масштабировать парсинг разных форматов, расширять pipeline обработки,
    логировать и анализировать качество данных и пайплайна.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        hooks_config: Optional[Dict[str, Dict[str, List[Dict[str, Any]]]]] = None,
        min_chunk_length: int = 20,
        max_chunk_length: int = 50000,
    ):
        self.logger = logger or logging.getLogger("RAGFileUtils")
        self.hook_manager = HookManager(self.logger)
        self.hooks_config = hooks_config or {"pre": {}, "post": {}}
        self.manager = FileProcessorManager(
            logger=self.logger,
            hook_manager=self.hook_manager,
            hooks_config=self.hooks_config
        )
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length

        self._register_all_parsers()
        self._register_hooks_from_config(self.hooks_config)

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
        # hooks_config = {'pre': {'.txt': [{'hook': SomeHook, 'params': {...}}], ... }, 'post': {...}}
        for typ in ("pre", "post"):
            for ext, hooks in hooks_config.get(typ, {}).items():
                for hook_def in hooks:
                    hook_cls = hook_def["hook"]
                    params = hook_def.get("params", {})
                    hook = hook_cls(**params)
                    if typ == "pre":
                        self.hook_manager.register_pre_hook(hook, formats=ext if ext != "default" else None)
                    else:
                        self.hook_manager.register_post_hook(hook, formats=ext if ext != "default" else None)

    def extract_text(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Извлекает текст и метаданные из любого поддерживаемого файла.
        Возвращает dict с ключами text, success, error, meta.
        Применяет post-processing фильтрацию чанков по длине.
        """
        result = self.manager.extract_text(file_path, **kwargs)
        filtered_result, filter_stats = self._filter_by_chunk_length(result)
        if filter_stats["filtered"]:
            self.logger.info(
                f"File {file_path}: {filter_stats['filtered']} из {filter_stats['total']} чанков отфильтровано по длине."
            )
        result["meta"]["chunk_filter_stats"] = filter_stats
        return filtered_result

    def _filter_by_chunk_length(self, result: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Фильтрует текст по min/max длине чанка (разделитель — \n\n или \n).
        Возвращает обновлённый result и статистику фильтрации.
        """
        text = result.get("text", "")
        if not text:
            return result, {"filtered": 0, "total": 0, "min_len": self.min_chunk_length, "max_len": self.max_chunk_length}
        ext = result.get("meta", {}).get("file_type", "")
        chunks = []
        if ext in ("csv", "xlsx", "xls"):
            # Табличные — по строке
            chunks = text.split("\n")
        else:
            # Остальные — по абзацу
            chunks = [chunk for chunk in text.split("\n\n") if chunk.strip()]
        total = len(chunks)
        filtered = [
            chunk for chunk in chunks
            if self.min_chunk_length <= len(chunk.strip()) <= self.max_chunk_length
        ]
        filtered_count = total - len(filtered)
        filtered_text = "\n\n".join(filtered) if ext not in ("csv", "xlsx", "xls") else "\n".join(filtered)
        new_result = result.copy()
        new_result["text"] = filtered_text
        return new_result, {
            "filtered": filtered_count,
            "total": total,
            "min_len": self.min_chunk_length,
            "max_len": self.max_chunk_length
        }

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
        Логирует агрегированную статистику по удалённым чанкам, длинам, ошибкам.
        """
        search_path = str(Path(dir_path) / pattern)
        files = glob(search_path, recursive=recursive)
        if not files:
            self.logger.warning(f"No files found for pattern {pattern} in {dir_path}")
        results = []
        agg_stats = {
            "files": 0,
            "success": 0,
            "partial": 0,
            "errors": 0,
            "filtered_chunks": 0,
            "total_chunks": 0,
            "skipped": 0,
        }
        for file_path in files:
            res = self.extract_text(file_path, **kwargs)
            results.append(res)
            stat = res.get("meta", {}).get("chunk_filter_stats", {})
            agg_stats["files"] += 1
            agg_stats["filtered_chunks"] += stat.get("filtered", 0)
            agg_stats["total_chunks"] += stat.get("total", 0)
            if res.get("success"):
                agg_stats["success"] += 1
            elif res.get("meta", {}).get("partial_success"):
                agg_stats["partial"] += 1
            else:
                agg_stats["errors"] += 1
        self.logger.info(
            f"BATCH: {agg_stats['files']} файлов, успешно: {agg_stats['success']}, частично: {agg_stats['partial']}, "
            f"errors: {agg_stats['errors']}, фильтровано чанков: {agg_stats['filtered_chunks']}/{agg_stats['total_chunks']}"
        )
        return results

    def get_supported_extensions(self) -> List[str]:
        """Возвращает список всех поддерживаемых расширений файлов."""
        return self.manager.get_supported_extensions()

    def get_stats_from_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Агрегирует статистику по batch-результатам: success_rate, avg_length, lang_distribution, фильтр чанк-аналитика.
        """
        total = len(results)
        success = sum(1 for r in results if r.get("success"))
        partial = sum(1 for r in results if r.get("meta", {}).get("partial_success"))
        errors = [r.get("error") for r in results if not r.get("success")]
        lengths = [r.get("meta", {}).get("chars", 0) for r in results if r.get("success")]
        langs = {}
        filtered_chunks = sum(r.get("meta", {}).get("chunk_filter_stats", {}).get("filtered", 0) for r in results)
        total_chunks = sum(r.get("meta", {}).get("chunk_filter_stats", {}).get("total", 0) for r in results)
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
            "filtered_chunks": filtered_chunks,
            "total_chunks": total_chunks,
            "filter_rate": round(filtered_chunks/total_chunks, 3) if total_chunks else 0,
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
