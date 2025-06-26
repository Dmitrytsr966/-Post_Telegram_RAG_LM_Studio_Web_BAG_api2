import logging
import time
import requests
from typing import Optional, Dict, Union

class TelegramClient:
    TELEGRAM_API_URL = "https://api.telegram.org"

    def __init__(self, token: str, channel_id: str, config: dict):
        self.logger = logging.getLogger("TelegramClient")
        self.token = token
        self.channel_id = channel_id
        self.config = config

        self.api_base = f"{self.TELEGRAM_API_URL}/bot{self.token}"

        self.max_text_length = config.get("max_text_length", 4096)
        self.max_caption_length = config.get("max_caption_length", 1024)
        self.parse_mode = config.get("parse_mode", "HTML")
        self.disable_preview = config.get("disable_web_page_preview", True)
        self.retry_attempts = config.get("retry_attempts", 3)
        self.retry_delay = config.get("retry_delay", 2)

    @staticmethod
    def _telegram_code_units(text: str) -> int:
        """Подсчёт длины текста/caption по code units (UTF-16) — как в Telegram API."""
        return len(text.encode('utf-16-le')) // 2

    def send_text_message(self, text: str) -> bool:
        length = self._telegram_code_units(text)
        if length > self.max_text_length:
            self.logger.warning(f"Text message exceeds Telegram limits: {length} > {self.max_text_length} code units.")
            return False

        payload = {
            "chat_id": self.channel_id,
            "text": self.format_message(text),
            "parse_mode": self.parse_mode,
            "disable_web_page_preview": self.disable_preview
        }
        result = self._post_with_retry("sendMessage", json=payload)
        if not result:
            self.logger.error(f"Failed to send text message (length={length} code units).")
        return result

    def send_media_message(self, text: str, media_path: str) -> bool:
        length = self._telegram_code_units(text)
        if length > self.max_caption_length:
            self.logger.warning(f"Caption exceeds Telegram limits: {length} > {self.max_caption_length} code units.")
            return False

        media_type = self.get_media_type(media_path)
        method = {
            "photo": "sendPhoto",
            "video": "sendVideo",
            "document": "sendDocument"
        }.get(media_type)

        if not method:
            self.logger.error(f"Unsupported media format for file: {media_path}")
            return False

        try:
            with open(media_path, "rb") as file:
                files = {media_type: file}
                data = {
                    "chat_id": self.channel_id,
                    "caption": self.format_message(text),
                    "parse_mode": self.parse_mode
                }
                result = self._post_with_retry(method, data=data, files=files)
                if not result:
                    self.logger.error(f"Failed to send media message (caption length={length} code units).")
                return result
        except Exception as e:
            self.logger.exception(f"Failed to open or send media: {media_path}")
            return False

    def _post_with_retry(self, method: str, json: dict = None, data: dict = None, files: dict = None) -> bool:
        url = f"{self.api_base}/{method}"
        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = requests.post(url, json=json, data=data, files=files, timeout=10)
                if response.status_code == 200:
                    self.logger.info(f"[Telegram] {method} successful.")
                    return True
                else:
                    self._log_telegram_failure(response, method)
                    if response.status_code in {400, 403}:
                        return False
                    if response.status_code == 429:
                        retry_after = response.json().get("parameters", {}).get("retry_after", 5)
                        self.logger.warning(f"Rate limited. Retrying in {retry_after}s...")
                        time.sleep(retry_after)
                        continue
            except Exception as e:
                self.logger.warning(f"Attempt {attempt}/{self.retry_attempts} failed for {method}: {str(e)}")
            time.sleep(self.retry_delay)
        self.logger.error(f"All attempts failed for {method}.")
        return False

    def _log_telegram_failure(self, response: requests.Response, method: str):
        try:
            payload = response.json()
        except ValueError:
            payload = {"error": "Invalid JSON from Telegram"}
        self.logger.warning(
            f"Telegram API failure [{method}]: {response.status_code} - {payload}"
        )

    def retry_send_message(self, message_data: dict, max_retries: int = 3) -> bool:
        return self._post_with_retry("sendMessage", json=message_data)

    def format_message(self, text: str) -> str:
        # Для parse_mode HTML Telegram сам не требует экранирования стандартных тегов,
        # но в спорных случаях лучше экранировать спецсимволы вне тегов.
        if self.parse_mode == "HTML":
            return text  # Предполагается, что текст уже подготовлен валидатором.
        return text

    def escape_html(self, text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def validate_message_length(self, text: str, has_media: bool) -> bool:
        limit = self.max_caption_length if has_media else self.max_text_length
        return self._telegram_code_units(text) <= limit

    def get_media_type(self, file_path: str) -> Optional[str]:
        ext = file_path.lower().split('.')[-1]
        if ext in ["jpg", "jpeg", "png", "webp"]:
            return "photo"
        elif ext in ["mp4", "mov", "avi"]:
            return "video"
        elif ext in ["pdf", "docx", "txt", "zip"]:
            return "document"
        return None

    def handle_telegram_errors(self, error: Exception) -> bool:
        self.logger.error(f"Handled Telegram error: {str(error)}")
        return False

    def check_bot_permissions(self) -> Dict[str, any]:
        try:
            resp = requests.get(f"{self.api_base}/getMe", timeout=10)
            data = resp.json()
            self.logger.info(f"Bot identity: {data}")
            return data
        except Exception:
            self.logger.error("Failed to fetch bot info.", exc_info=True)
            return {}

    def get_channel_info(self) -> Dict[str, any]:
        try:
            resp = requests.get(f"{self.api_base}/getChat", params={"chat_id": self.channel_id}, timeout=10)
            return resp.json()
        except Exception:
            self.logger.warning("Failed to retrieve channel info.", exc_info=True)
            return {}

    def get_send_stats(self) -> dict:
        return {
            "channel_id": self.channel_id,
            "token_hash": hash(self.token),
            "parse_mode": self.parse_mode,
            "retry_attempts": self.retry_attempts
        }
