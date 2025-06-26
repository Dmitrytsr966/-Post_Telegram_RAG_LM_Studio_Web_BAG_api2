import logging
import pandas as pd
from pathlib import Path

def extract_text(file_path: str, **kwargs) -> dict:
    result = {
        "text": "",
        "success": False,
        "error": None,
        "cleaned": False,
        "meta": {
            "file_path": file_path,
            "file_type": "csv",
            "parser": "rag_csv",
            "lines": 0,
            "chars": 0,
            "percent_empty": 0.0,
        }
    }
    try:
        encoding = kwargs.get("encoding", "utf-8")
        df = pd.read_csv(file_path, encoding=encoding)
        orig_rows = len(df)
        df = df.dropna(how='all')
        # Remove rows that are all empty or whitespace
        def is_useless_row(row):
            return all((str(cell).strip() == "" or str(cell).strip() == "nan") for cell in row)
        cleaned_df = df[~df.apply(is_useless_row, axis=1)]
        percent_empty = 100.0 * (orig_rows - len(cleaned_df)) / (orig_rows or 1)
        text = cleaned_df.to_string(index=False)
        result["text"] = text.strip()
        result["success"] = True
        result["cleaned"] = (len(cleaned_df) != orig_rows)
        result["meta"]["lines"] = len(cleaned_df)
        result["meta"]["chars"] = len(result["text"])
        result["meta"]["percent_empty"] = percent_empty
    except Exception as e:
        result["error"] = str(e)
        logging.getLogger("rag_csv").warning(f"CSV extraction failed for {file_path}: {e}")
    return result
