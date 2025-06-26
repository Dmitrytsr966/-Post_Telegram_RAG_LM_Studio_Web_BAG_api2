import logging
from pathlib import Path

def extract_text(file_path: str, **kwargs) -> dict:
    result = {
        "text": "",
        "success": False,
        "error": None,
        "cleaned": False,
        "meta": {
            "file_path": file_path,
            "file_type": "txt",
            "parser": "rag_txt",
            "lines": 0,
            "chars": 0,
        }
    }
    try:
        p = Path(file_path)
        text = p.read_text(encoding=kwargs.get("encoding", "utf-8"))
        cleaned_text = text.strip()
        result["text"] = cleaned_text
        result["success"] = True
        result["cleaned"] = (cleaned_text != text)
        result["meta"]["lines"] = cleaned_text.count('\n') + 1
        result["meta"]["chars"] = len(cleaned_text)
    except Exception as e:
        result["error"] = str(e)
        logging.getLogger("rag_txt").warning(f"TXT extraction failed for {file_path}: {e}")
    return result
