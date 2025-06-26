import logging
from pathlib import Path
import re

def extract_text(file_path: str, **kwargs) -> dict:
    result = {
        "text": "",
        "success": False,
        "error": None,
        "cleaned": False,
        "meta": {
            "file_path": file_path,
            "file_type": "md",
            "parser": "rag_markdown",
            "lines": 0,
            "chars": 0,
        }
    }
    try:
        p = Path(file_path)
        text = p.read_text(encoding=kwargs.get("encoding", "utf-8"))
        # Remove code blocks and tables
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'\|.*\|', '', text)
        cleaned_text = re.sub(r'#.*', '', text)
        cleaned_text = "\n".join(line.strip() for line in cleaned_text.splitlines() if line.strip())
        result["text"] = cleaned_text
        result["success"] = True
        result["cleaned"] = True
        result["meta"]["lines"] = cleaned_text.count('\n') + 1
        result["meta"]["chars"] = len(cleaned_text)
    except Exception as e:
        result["error"] = str(e)
        logging.getLogger("rag_markdown").warning(f"Markdown extraction failed for {file_path}: {e}")
    return result
