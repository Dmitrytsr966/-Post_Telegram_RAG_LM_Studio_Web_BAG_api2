import logging
import json
from pathlib import Path

def extract_text(file_path: str, **kwargs) -> dict:
    result = {
        "text": "",
        "success": False,
        "error": None,
        "cleaned": False,
        "meta": {
            "file_path": file_path,
            "file_type": "json",
            "parser": "rag_json",
            "lines": 0,
            "chars": 0,
        }
    }
    try:
        p = Path(file_path)
        data = json.loads(p.read_text(encoding=kwargs.get("encoding", "utf-8")))
        import pandas as pd
        # Convert to flat text via pandas if possible
        if isinstance(data, list):
            df = pd.DataFrame(data)
            text = df.to_string(index=False)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
            text = df.to_string(index=False)
        else:
            text = str(data)
        cleaned_text = text.strip()
        result["text"] = cleaned_text
        result["success"] = True
        result["cleaned"] = True
        result["meta"]["lines"] = cleaned_text.count('\n') + 1
        result["meta"]["chars"] = len(cleaned_text)
    except Exception as e:
        result["error"] = str(e)
        logging.getLogger("rag_json").warning(f"JSON extraction failed for {file_path}: {e}")
    return result
