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
            "file_type": "docx",
            "parser": "rag_docx",
            "lines": 0,
            "chars": 0,
            "used_fallback": False,
        }
    }
    try:
        import docx
        doc = docx.Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs)
        cleaned_text = text.strip()
        result["text"] = cleaned_text
        result["success"] = True
        result["meta"]["lines"] = cleaned_text.count('\n') + 1
        result["meta"]["chars"] = len(cleaned_text)
    except Exception as e:
        try:
            import textract
            text = textract.process(file_path).decode("utf-8")
            cleaned_text = text.strip()
            result["text"] = cleaned_text
            result["success"] = True
            result["meta"]["used_fallback"] = True
            result["meta"]["lines"] = cleaned_text.count('\n') + 1
            result["meta"]["chars"] = len(cleaned_text)
            logging.getLogger("rag_docx").info(f"DOCX extraction via textract fallback for {file_path}")
        except Exception as e2:
            result["error"] = f"docx: {str(e)}; textract: {str(e2)}"
            logging.getLogger("rag_docx").warning(f"DOCX extraction failed for {file_path}: {result['error']}")
    return result
