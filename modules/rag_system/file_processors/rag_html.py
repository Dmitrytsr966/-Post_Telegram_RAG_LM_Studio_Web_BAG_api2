import logging
from pathlib import Path
from bs4 import BeautifulSoup

def extract_text(file_path: str, **kwargs) -> dict:
    result = {
        "text": "",
        "success": False,
        "error": None,
        "cleaned": False,
        "meta": {
            "file_path": file_path,
            "file_type": "html",
            "parser": "rag_html",
            "lines": 0,
            "chars": 0,
        }
    }
    try:
        p = Path(file_path)
        text = p.read_text(encoding=kwargs.get("encoding", "utf-8"))
        soup = BeautifulSoup(text, "lxml")
        for t in soup(["script", "style"]):
            t.decompose()
        cleaned_text = soup.get_text(separator="\n")
        cleaned_text = "\n".join(line.strip() for line in cleaned_text.splitlines() if line.strip())
        result["text"] = cleaned_text
        result["success"] = True
        result["cleaned"] = True
        result["meta"]["lines"] = cleaned_text.count('\n') + 1
        result["meta"]["chars"] = len(cleaned_text)
    except Exception as e:
        result["error"] = str(e)
        logging.getLogger("rag_html").warning(f"HTML extraction failed for {file_path}: {e}")
    return result
