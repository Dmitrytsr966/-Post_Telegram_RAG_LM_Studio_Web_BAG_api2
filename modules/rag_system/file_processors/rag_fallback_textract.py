import logging

def extract_text(file_path: str, **kwargs) -> dict:
    result = {
        "text": "",
        "success": False,
        "error": None,
        "cleaned": False,
        "meta": {
            "file_path": file_path,
            "file_type": "fallback",
            "parser": "rag_fallback_textract",
            "used_fallback": True,
        }
    }
    try:
        import textract
        text = textract.process(file_path).decode("utf-8")
        cleaned_text = text.strip()
        result["text"] = cleaned_text
        result["success"] = True
        result["meta"]["lines"] = cleaned_text.count('\n') + 1
        result["meta"]["chars"] = len(cleaned_text)
    except Exception as e:
        result["error"] = str(e)
        logging.getLogger("rag_fallback_textract").warning(f"Textract fallback failed for {file_path}: {e}")
    return result
