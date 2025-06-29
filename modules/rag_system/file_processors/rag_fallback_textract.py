import logging
from typing import Optional, Callable, Dict, Any, List

def extract_text(
    file_path: str,
    extra_hooks: Optional[List[Callable]] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Заглушка fallback-парсера, если textract не установлен.
    Возвращает ошибку, информируя, что textract не доступен.
    """
    logger = logger or logging.getLogger("rag_fallback_textract")
    meta = {
        "file_path": file_path,
        "file_type": "fallback",
        "parser": "rag_fallback_textract",
        "used_fallback": True,
        "cleaned": False,
        "lines": 0,
        "chars": 0,
        "hooks_applied": [],
        "error": "textract is not installed or not available"
    }
    logger.warning(
        f"Textract fallback is not available (textract not installed). File: {file_path}"
    )
    return {
        "text": "",
        "success": False,
        "error": "textract is not installed or not available",
        "cleaned": False,
        "meta": meta
    }
