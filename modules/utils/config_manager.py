import json
import os
import logging
from typing import Optional, Any, Dict

class ConfigManager:
    """
    Менеджер конфигураций для всей системы автопостинга с RAG и LM Studio.
    Гарантирует: корректную загрузку, подробную валидацию и безопасный доступ к параметрам.
    """

    def __init__(self, config_path: str = "config/config.json"):
        self.logger = logging.getLogger("ConfigManager")
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """
        Загружает конфиг из JSON-файла. В случае ошибки (отсутствие, синтаксис) — критический лог и exit.
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.logger.info(f"Loaded config from {self.config_path}")
            return config
        except FileNotFoundError:
            self.logger.critical(f"Config file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.critical(f"Failed to parse config.json: {e}")
            raise

    def validate_config(self) -> bool:
        """
        Проверяет наличие всех обязательных секций и ключей. Логирует каждую причину.
        Возвращает: True — если всё ок, иначе False.
        """
        errors = []
        config = self.config

        # Обязательные секции
        required_sections = ["lm_studio", "rag", "telegram", "serper", "processing", "paths"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: '{section}'")

        # Проверка ключей для lm_studio
        if "lm_studio" in config:
            for key in ["base_url", "model", "max_tokens", "temperature", "timeout"]:
                if key not in config["lm_studio"]:
                    errors.append(f"Missing key '{key}' in section 'lm_studio'")

        # Проверка ключей для rag
        if "rag" in config:
            for key in ["embedding_model", "chunk_size", "chunk_overlap", "max_context_length", "media_context_length", "similarity_threshold"]:
                if key not in config["rag"]:
                    errors.append(f"Missing key '{key}' in section 'rag'")

        # Проверка ключей telegram
        if "telegram" in config:
            for key in ["post_interval", "max_retries"]:
                if key not in config["telegram"]:
                    errors.append(f"Missing key '{key}' in section 'telegram'")

        # serper
        if "serper" in config:
            for key in ["results_limit"]:
                if key not in config["serper"]:
                    errors.append(f"Missing key '{key}' in section 'serper'")

        # processing
        if "processing" in config:
            for key in ["batch_size", "max_file_size_mb"]:
                if key not in config["processing"]:
                    errors.append(f"Missing key '{key}' in section 'processing'")

        # paths
        if "paths" in config:
            for key in ["media_dir", "prompt_folders", "data_dir", "processed_topics_file"]:
                if key not in config["paths"]:
                    errors.append(f"Missing key '{key}' in section 'paths'")

        if errors:
            for err in errors:
                self.logger.critical(f"Config validation error: {err}")
            return False
        return True

    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """
        Позволяет получать значение по "пути" через точку, например: 'lm_studio.base_url'
        Если не найдено — возвращает default.
        """
        keys = key_path.split(".")
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            self.logger.warning(f"Config key not found: {key_path}, using default: {default}")
            return default

    def get_telegram_token(self) -> str:
        """Читает токен Telegram-бота из config/telegram_token.txt"""
        token_file = os.path.join("config", "telegram_token.txt")
        try:
            with open(token_file, "r", encoding="utf-8") as f:
                token = f.read().strip()
            if not token:
                self.logger.critical("Telegram token file is empty!")
                raise ValueError("Empty Telegram token")
            return token
        except Exception as e:
            self.logger.critical(f"Failed to read Telegram token: {e}")
            raise

    def get_telegram_channel_id(self) -> str:
        """Читает ID канала Telegram из config/telegram_channel.txt"""
        channel_file = os.path.join("config", "telegram_channel.txt")
        try:
            with open(channel_file, "r", encoding="utf-8") as f:
                channel_id = f.read().strip()
            if not channel_id:
                self.logger.critical("Telegram channel ID file is empty!")
                raise ValueError("Empty Telegram channel ID")
            return channel_id
        except Exception as e:
            self.logger.critical(f"Failed to read Telegram channel ID: {e}")
            raise

    def get_lm_studio_config(self) -> dict:
        return self.config.get("lm_studio", {})

    def get_rag_config(self) -> dict:
        return self.config.get("rag", {})

    def get_serper_api_key(self) -> Optional[str]:
        """Пробует взять serper API ключ из переменной окружения или файла."""
        api_key = os.environ.get("SERPER_API_KEY")
        if api_key:
            return api_key
        # Можно добавить вариант с файлом
        key_file = os.path.join("config", "serper_api_key.txt")
        if os.path.exists(key_file):
            with open(key_file, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
            if api_key:
                return api_key
        # Fallback: попробовать из конфига
        return self.config.get("serper", {}).get("api_key")

    def get_all_config(self) -> dict:
        """Возвращает полный конфиг (для отладки, без секретных полей)."""
        safe_config = self.config.copy()
        # Можно тут удалить/заменить чувствительные данные, если они есть
        return safe_config

    def save_config(self) -> None:
        """Сохраняет текущий конфиг обратно в файл."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            self.logger.info("Config saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")

    def reload_config(self) -> None:
        """Перечитывает конфиг с диска."""
        self.config = self._load_config()
        self.logger.info("Config reloaded.")
