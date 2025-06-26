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
            "file_type": "pdf",
            "parser": "rag_pdf",
            "lines": 0,
            "chars": 0,
            "used_fallback": False,
        }
    }
    try:
        import pdfplumber
        text_blocks = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_blocks.append(page_text)
        text = "\n".join(block for block in text_blocks if block and block.strip())
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("pdfplumber extracted no text")
        result["text"] = cleaned_text
        result["success"] = True
        result["meta"]["lines"] = cleaned_text.count('\n') + 1
        result["meta"]["chars"] = len(cleaned_text)
    except Exception as e:
        try:
            import textract
            text = textract.process(file_path).decode("utf-8")
            cleaned_text = text.strip()
            if not cleaned_text:
                raise ValueError("textract extracted no text")
            result["text"] = cleaned_text
            result["success"] = True
            result["meta"]["used_fallback"] = True
            result["meta"]["lines"] = cleaned_text.count('\n') + 1
            result["meta"]["chars"] = len(cleaned_text)
            logging.getLogger("rag_pdf").info(f"PDF extraction via textract fallback for {file_path}")
        except Exception as e2:
            result["error"] = f"pdfplumber: {str(e)}; textract: {str(e2)}"
            logging.getLogger("rag_pdf").warning(f"PDF extraction failed for {file_path}: {result['error']}")
    return result
