import logging
import time
import requests
import random
import re
import json
from typing import Optional, Dict, Union, List

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

        # Кнопки по умолчанию для каждого поста
        self.default_buttons = [
            [
                {
                    "text": "🎀 Работа моделью 🎀",
                    "url": "https://t.me/Flagman_tm_bot"
                }
            ]
        ]

    @staticmethod
    def _telegram_code_units(text: str) -> int:
        """Подсчёт длины текста/caption по code units (UTF-16) — как в Telegram API."""
        return len(text.encode('utf-16-le')) // 2

    def _replace_casino_words(self, text: str) -> str:
        """Заменяет слова 'казино' и 'casino' на варианты с замещением 'а' и 'о' буквами из другого алфавита."""
        def replace_russian_casino(match):
            word = match.group(0)
            # Для русского слова заменяем 'а' и 'о' на английские
            result = word
            # Заменяем все 'а' на 'a'
            result = result.replace('а', 'a').replace('А', 'A')
            # Заменяем все 'о' на 'o'  
            result = result.replace('о', 'o').replace('О', 'O')
            return result

        def replace_english_casino(match):
            word = match.group(0)
            # Для английского слова заменяем 'a' и 'o' на русские
            result = word
            # Заменяем все 'a' на 'а'
            result = result.replace('a', 'а').replace('A', 'А')
            # Заменяем все 'o' на 'о'
            result = result.replace('o', 'о').replace('O', 'О')
            return result

        # Заменяем русские варианты (включая через дефис)
        # Используем простые паттерны без проблемных lookbehind переменной ширины
        text = re.sub(r'\bказино\b', replace_russian_casino, text, flags=re.IGNORECASE)
        text = re.sub(r'(?<!\w)казино(?!\w)', replace_russian_casino, text, flags=re.IGNORECASE)
        text = re.sub(r'(?<=-)казино\b', replace_russian_casino, text, flags=re.IGNORECASE)
        text = re.sub(r'\bказино(?=-)', replace_russian_casino, text, flags=re.IGNORECASE)
        
        # Заменяем английские варианты (включая через дефис)
        text = re.sub(r'\bcasino\b', replace_english_casino, text, flags=re.IGNORECASE)
        text = re.sub(r'(?<!\w)casino(?!\w)', replace_english_casino, text, flags=re.IGNORECASE)
        text = re.sub(r'(?<=-)casino\b', replace_english_casino, text, flags=re.IGNORECASE)
        text = re.sub(r'\bcasino(?=-)', replace_english_casino, text, flags=re.IGNORECASE)
        
        return text

    def _check_minimum_length(self, text: str) -> bool:
        """Проверяет, что текст содержит не менее 712 символов с учетом пробелов."""
        return len(text) >= 712

    def send_text_message(self, text: str, buttons: Optional[List] = None) -> bool:
        # Проверка минимальной длины
        if not self._check_minimum_length(text):
            self.logger.warning(f"Text message too short: {len(text)} characters (minimum 712 required).")
            return False

        # Замена слов казино/casino
        text = self._replace_casino_words(text)

        length = self._telegram_code_units(text)
        if length > self.max_text_length:
            self.logger.warning(f"Text message exceeds Telegram limits: {length} > {self.max_text_length} code units.")
            return False

        reply_markup = self._build_inline_keyboard_markup(buttons or self.default_buttons)

        payload = {
            "chat_id": self.channel_id,
            "text": self.format_message(text),
            "parse_mode": self.parse_mode,
            "disable_web_page_preview": self.disable_preview,
            "reply_markup": reply_markup
        }
        result = self._post_with_retry("sendMessage", json=payload)
        if not result:
            self.logger.error(f"Failed to send text message (length={length} code units).")
        return result

    def send_media_message(self, text: str, media_path: str, buttons: Optional[List] = None) -> bool:
        # Проверка минимальной длины
        if not self._check_minimum_length(text):
            self.logger.warning(f"Media caption too short: {len(text)} characters (minimum 712 required).")
            return False

        # Замена слов казино/casino
        text = self._replace_casino_words(text)

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

        reply_markup = self._build_inline_keyboard_markup(buttons or self.default_buttons)

        try:
            with open(media_path, "rb") as file:
                files = {media_type: file}
                data = {
                    "chat_id": self.channel_id,
                    "caption": self.format_message(text),
                    "parse_mode": self.parse_mode,
                    "reply_markup": reply_markup
                }
                result = self._post_with_retry(method, data=data, files=files)
                if not result:
                    self.logger.error(f"Failed to send media message (caption length={length} code units).")
                return result
        except Exception as e:
            self.logger.exception(f"Failed to open or send media: {media_path}")
            return False

    def _build_inline_keyboard_markup(self, buttons: List) -> str:
        """Преобразует список кнопок в JSON-строку, подходящую для reply_markup Telegram.
        Поддерживает как одномерные списки (кнопки в столбик), так и двумерные (кнопки в строки и столбцы)."""
        
        # Проверяем, является ли это двумерным массивом (новый формат)
        if buttons and isinstance(buttons[0], list):
            # Новый формат: список строк, каждая строка содержит список кнопок
            keyboard = []
            for row in buttons:
                keyboard_row = []
                for btn in row:
                    keyboard_row.append({
                        "text": btn.get("text", ""),
                        "url": btn.get("url", "")
                    })
                keyboard.append(keyboard_row)
        else:
            # Старый формат: каждая кнопка — отдельная строка (каждая — отдельный ряд)
            keyboard = [[{
                "text": btn.get("text", ""),
                "url": btn.get("url", "")
            }] for btn in buttons]
        
        return json.dumps({"inline_keyboard": keyboard}, ensure_ascii=False)

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
        if self.parse_mode == "HTML":
            return text  # Предполагается, что текст уже подготовлен валидатором.
        return text

    def escape_html(self, text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def validate_message_length(self, text: str, has_media: bool) -> bool:
        # Проверка минимальной длины
        if not self._check_minimum_length(text):
            return False
        
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
