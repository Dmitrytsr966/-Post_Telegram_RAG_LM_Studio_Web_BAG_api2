import logging
from pathlib import Path
from typing import Callable, List, Dict, Optional, Any, Tuple
from .hook_manager import HookManager, HookResult

class FileProcessorManager:
    """
    Гибкий менеджер парсеров и хуков для любых файлов.
    Поддержка централизованной конфигурации pre/post хуков с параметрами, fallback, аналитики, batch-режимов.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        hook_manager: Optional[HookManager] = None,
        hooks_config: Optional[Dict[str, Dict[str, List[Dict[str, Any]]]]] = None
    ):
        self.logger = logger or logging.getLogger("FileProcessorManager")
        self.parsers: Dict[str, Callable] = {}
        self.fallbacks: List[Callable] = []
        self.hook_manager = hook_manager or HookManager(self.logger)
        # hooks_config: {'pre': {'.csv': [ {'hook': HookClass, 'params': {...}}, ... ], ... }, 'post': {...} }
        self.hooks_config = hooks_config or {"pre": {}, "post": {}}
        self._instantiated_hooks_cache = {"pre": {}, "post": {}}

    def register_parser(self, extensions, parser_func):
        if isinstance(extensions, str):
            extensions = [extensions]
        for ext in extensions:
            self.parsers[ext.lower()] = parser_func

    def register_fallback(self, parser_func):
        self.fallbacks.append(parser_func)

    def _get_hooks_for_extension(self, ext: str, hook_type: str) -> List[Callable]:
        """
        Возвращает список инстансов хуков для расширения и типа (pre/post).
        Использует кэш для предотвращения лишнего инстанцирования.
        """
        ext = ext.lower()
        config = self.hooks_config.get(hook_type, {})
        hooks = config.get(ext, []) + config.get("default", [])
        cache = self._instantiated_hooks_cache[hook_type].setdefault(ext, [])
        if not cache or len(cache) != len(hooks):
            cache.clear()
            for hook_info in hooks:
                hook_class = hook_info.get('hook')
                params = hook_info.get('params', {})
                try:
                    hook_instance = hook_class(**params)
                    cache.append(hook_instance)
                except Exception as e:
                    self.logger.error(f"Failed to instantiate {hook_type}-hook {hook_class}: {e}")
        return list(cache)

    def _apply_hooks(
        self,
        text: str,
        meta: dict,
        ext: str,
        hooks: List[Callable],
        hook_type: str
    ) -> Tuple[str, List[str], List[dict]]:
        """
        Применяет цепочку хуков по очереди, собирает HookResult по каждому.
        Возвращает text, цепочку имён, список dict-результатов.
        """
        chain = []
        results = []
        for hook in hooks:
            hook_name = type(hook).__name__
            old_text = text
            try:
                text = hook(text, meta)
                summary = None
                if hasattr(hook, "summary"):
                    summary = hook.summary(old_text, text)
                result = HookResult(
                    hook_name=hook_name,
                    old_text=old_text,
                    new_text=text,
                    params=getattr(hook, "params", {}),
                    summary=summary
                )
            except Exception as e:
                self.logger.error(f"Failed to apply {hook_type}-hook {hook_name}: {e}")
                result = HookResult(
                    hook_name=hook_name,
                    old_text=old_text,
                    new_text=old_text,
                    params=getattr(hook, "params", {}),
                    error=str(e)
                )
                text = old_text  # fail-safe
            chain.append(hook_name)
            results.append(result.as_dict())
        if chain:
            self.logger.info(
                f"{hook_type.capitalize()}-hooks applied ({ext}): {chain} | Stats: {results}"
            )
        return text, chain, results

    def extract_text(self, file_path: str, **kwargs) -> dict:
        ext = Path(file_path).suffix.lower()
        parser = self.parsers.get(ext)
        meta = {
            "file_path": file_path,
            "file_type": ext[1:] if ext else "unknown",
            "parser_chain": [],
            "pre_hook_chain": [],
            "post_hook_chain": [],
            "pre_hook_stats": [],
            "post_hook_stats": [],
        }
        text = ""
        # 1. Pre-hooks
        pre_hooks = self._get_hooks_for_extension(ext, "pre")
        if pre_hooks:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception as e:
                self.logger.error(f"Failed to read file for pre-hooks: {file_path}: {e}")
                return {"text": "", "success": False, "error": f"Pre-hooks file read error: {e}", "meta": meta}
            text, pre_hook_chain, pre_hook_stats = self._apply_hooks(text, meta, ext, pre_hooks, "pre")
            meta["pre_hook_chain"] = pre_hook_chain
            meta["pre_hook_stats"] = pre_hook_stats
        else:
            text = file_path  # для парсеров, ожидающих путь

        # 2. Парсер
        if parser:
            try:
                # Если pre-хуки применялись, передаем text, иначе путь
                result = parser(text if pre_hooks else file_path, **kwargs)
            except Exception as e:
                self.logger.error(f"Parser exception for {file_path}: {e}")
                result = {"text": "", "success": False, "error": str(e), "meta": {}}
            text = result.get("text", "")
            meta.update(result.get("meta", {}))
            meta["parser_chain"].append(parser.__name__)
            error = result.get("error")
            # 3. Post-hooks
            post_hooks = self._get_hooks_for_extension(ext, "post")
            if post_hooks:
                text, post_hook_chain, post_hook_stats = self._apply_hooks(text, meta, ext, post_hooks, "post")
                meta["post_hook_chain"] = post_hook_chain
                meta["post_hook_stats"] = post_hook_stats
            # Partial success
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

        # 4. Fallbacks
        for fallback in self.fallbacks:
            try:
                result = fallback(file_path, **kwargs)
            except Exception as e:
                self.logger.error(f"Fallback parser exception for {file_path}: {e}")
                result = {"text": "", "success": False, "error": str(e), "meta": {}}
            text = result.get("text", "")
            meta.update(result.get("meta", {}))
            meta["parser_chain"].append(fallback.__name__)
            error = result.get("error")
            post_hooks = self._get_hooks_for_extension(ext, "post")
            if post_hooks:
                text, post_hook_chain, post_hook_stats = self._apply_hooks(text, meta, ext, post_hooks, "post")
                meta["post_hook_chain"] = post_hook_chain
                meta["post_hook_stats"] = post_hook_stats
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
