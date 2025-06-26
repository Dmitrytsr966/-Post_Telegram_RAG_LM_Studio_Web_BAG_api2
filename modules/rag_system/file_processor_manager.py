import logging
from pathlib import Path
from typing import Callable, List, Dict, Optional
from .hook_manager import HookManager

class FileProcessorManager:
    """
    Мощный менеджер парсеров и хуков для обработки любых файлов.
    Поддержка цепочек хуков, fallback, аналитики, batch-режимов.
    """
    def __init__(self, logger=None, hook_manager: Optional[HookManager]=None):
        self.logger = logger or logging.getLogger("FileProcessorManager")
        self.parsers: Dict[str, Callable] = {}
        self.fallbacks: List[Callable] = []
        self.hook_manager = hook_manager or HookManager(self.logger)

    def register_parser(self, extensions, parser_func):
        if isinstance(extensions, str):
            extensions = [extensions]
        for ext in extensions:
            self.parsers[ext.lower()] = parser_func

    def register_fallback(self, parser_func):
        self.fallbacks.append(parser_func)

    def extract_text(self, file_path: str, **kwargs) -> dict:
        ext = Path(file_path).suffix.lower()
        parser = self.parsers.get(ext)
        meta = {
            "file_path": file_path,
            "file_type": ext[1:] if ext else "unknown",
            "parser_chain": [],
            "pre_hook_chain": [],
            "post_hook_chain": [],
        }
        text, pre_hook_chain, post_hook_chain = "", [], []

        # Применяем pre-хуки (если есть)
        # (опционально: можно реализовать для предобработки файла/пути)

        if parser:
            # Вызов парсера
            result = parser(file_path, **kwargs)
            text = result.get("text", "")
            meta.update(result.get("meta", {}))
            meta["parser_chain"].append(parser.__name__)
            error = result.get("error")
            # Пост-хуки (на обработанный текст)
            text, post_hook_chain = self.hook_manager.apply_post_hooks(text, meta, ext)
            meta["post_hook_chain"] = post_hook_chain
            # Partial success detection
            if error and text:
                meta["partial_success"] = True
                meta["partial_reason"] = error
                self.logger.warning(f"Partial success on {file_path}: {error}")
                return {
                    "text": "",
                    "success": False,
                    "error": f"Partial success: {error}",
                    "meta": meta,
                }
            elif result.get("success"):
                return {
                    "text": text,
                    "success": True,
                    "error": None,
                    "meta": meta,
                }
            self.logger.warning(f"Primary parser failed for {file_path}. Trying fallbacks.")

        # Fallbacks
        for fallback in self.fallbacks:
            result = fallback(file_path, **kwargs)
            text = result.get("text", "")
            meta.update(result.get("meta", {}))
            meta["parser_chain"].append(fallback.__name__)
            error = result.get("error")
            text, post_hook_chain = self.hook_manager.apply_post_hooks(text, meta, ext)
            meta["post_hook_chain"] = post_hook_chain
            if error and text:
                meta["partial_success"] = True
                meta["partial_reason"] = error
                self.logger.warning(f"Partial success (fallback) on {file_path}: {error}")
                return {
                    "text": "",
                    "success": False,
                    "error": f"Partial success: {error}",
                    "meta": meta,
                }
            elif result.get("success"):
                self.logger.info(f"Used fallback parser for {file_path}")
                return {
                    "text": text,
                    "success": True,
                    "error": None,
                    "meta": meta,
                }
        # Не удалось обработать
        meta["parser_chain"] = meta.get("parser_chain", [])
        self.logger.error(f"All parsers failed for {file_path}")
        return {
            "text": "",
            "success": False,
            "error": f"No parser succeeded for {file_path}",
            "meta": meta,
        }

    def extract_text_batch(self, files: List[str], skip_partial=True, **kwargs) -> List[dict]:
        """
        Batch-обработка списка файлов.
        skip_partial — если True, partial_success файлы не включаются в результат.
        """
        results = []
        for file_path in files:
            result = self.extract_text(file_path, **kwargs)
            if result.get("success"):
                results.append(result)
            elif not skip_partial and result["meta"].get("partial_success"):
                results.append(result)
            else:
                self.logger.info(f"Skipped file (failure or partial): {file_path}")
        return results

    def get_supported_extensions(self):
        return list(self.parsers.keys())
