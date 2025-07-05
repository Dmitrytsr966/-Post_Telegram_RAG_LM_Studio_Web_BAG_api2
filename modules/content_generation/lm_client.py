import logging
import requests
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from modules.content_generation.content_validator import ContentValidator

class FreeGPT4Client:
    LM_MAX_TOTAL_CHARS = 20000
    TELEGRAM_LIMIT = 4096

    def __init__(self, url: str, model: str, config: Dict[str, Any]):
        """
        url: URL до OpenAI-compatible endpoint (например, http://localhost:1337/v1/chat/completions)
        model: имя модели (например, gpt-4, deepseek-r1 и т.д.)
        config: словарь с параметрами (temperature, max_tokens, history_limit, system_message и т.д.)
        """
        self.url = url.rstrip("/")
        self.model = model
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.7)
        self.timeout = config.get("timeout", 60)
        self.history_limit = config.get("history_limit", 3)
        self.system_message = config.get("system_message", None)
        self.top_p = config.get("top_p", None)
        self.top_k = config.get("top_k", None)
        self.logger = logging.getLogger("FreeGPT4Client")
        self.history: List[Dict[str, str]] = []
        self.content_validator = ContentValidator(config=config)
        self.log_dir_success = "logs/freegpt4/success"
        self.log_dir_failed = "logs/freegpt4/failed"
        self.log_dir_prompts = "logs/freegpt4/prompts"
        os.makedirs(self.log_dir_success, exist_ok=True)
        os.makedirs(self.log_dir_failed, exist_ok=True)
        os.makedirs(self.log_dir_prompts, exist_ok=True)
        self._validate_config()

    def _validate_config(self):
        assert isinstance(self.max_tokens, int) and self.max_tokens > 0, "max_tokens must be positive integer"
        assert isinstance(self.LM_MAX_TOTAL_CHARS, int) and self.LM_MAX_TOTAL_CHARS > 1000, "LM_MAX_TOTAL_CHARS must be > 1000"
        assert isinstance(self.temperature, (float, int)), "temperature must be float"
        if self.top_p is not None:
            assert 0.0 <= self.top_p <= 1.0, "top_p must be in [0,1]"
        if self.top_k is not None:
            assert isinstance(self.top_k, int) and self.top_k >= 0, "top_k must be non-negative int"

    def clear_conversation_history(self):
        self.history = []
        self.logger.debug("FreeGPT4Client: conversation history cleared.")

    def add_to_history(self, user_message: str, bot_message: str):
        if user_message and isinstance(user_message, str) and user_message.strip():
            self.history.append({"role": "user", "content": user_message})
        if bot_message and isinstance(bot_message, str) and bot_message.strip():
            self.history.append({"role": "assistant", "content": bot_message})
        if self.history_limit > 0 and len(self.history) > self.history_limit * 2:
            self.history = self.history[-self.history_limit * 2:]

    def _clean_history(self) -> List[Dict[str, str]]:
        clean = []
        for m in self.history[-self.history_limit * 2:]:
            if (
                isinstance(m, dict)
                and m.get("role") in {"user", "assistant", "system"}
                and isinstance(m.get("content"), str)
                and m["content"].strip()
                and "nan" not in m["content"]
            ):
                clean.append(m)
            else:
                self.logger.warning(f"Skipping invalid message in LLM history: {m}")
        return clean

    def _truncate_context_for_llm(self, prompt_template: str, topic: str, context: str) -> str:
        replacements = {
            "{TOPIC}": topic.strip(),
            "{CONTEXT}": "CONTEXT_PLACEHOLDER",
        }
        prompt_wo_context = prompt_template
        for key, value in replacements.items():
            prompt_wo_context = prompt_wo_context.replace(key, value)
        prompt_wo_context_len = len(prompt_wo_context.replace("CONTEXT_PLACEHOLDER", ""))

        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.extend(self._clean_history())
        static_chars = sum(len(m["content"]) for m in messages) + prompt_wo_context_len
        available = self.LM_MAX_TOTAL_CHARS - static_chars
        if available <= 0:
            self.logger.warning(f"No room for context: static_chars={static_chars} > limit={self.LM_MAX_TOTAL_CHARS}")
            return ""
        context = context.strip()
        if len(context) > available:
            self.logger.warning(f"Context too long for LLM ({len(context)} > {available}), truncating context")
            context = context[:available]
        return context

    def _build_messages(self, prompt_template: str, topic: str, context: str) -> List[Dict[str, str]]:
        context = self._truncate_context_for_llm(prompt_template, topic, context)
        replacements = {
            "{TOPIC}": topic.strip(),
            "{CONTEXT}": context,
        }
        prompt = prompt_template
        for key, value in replacements.items():
            prompt = prompt.replace(key, value)
        prompt = prompt.replace("nan", "").strip()

        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.extend(self._clean_history())
        messages.append({"role": "user", "content": prompt})

        total_chars = sum(len(m["content"]) for m in messages)
        if total_chars > self.LM_MAX_TOTAL_CHARS:
            self.logger.warning(f"Total LLM payload too long ({total_chars} > {self.LM_MAX_TOTAL_CHARS}), trimming prompt/history")
            excess = total_chars - self.LM_MAX_TOTAL_CHARS
            if len(messages[-1]["content"]) > excess:
                messages[-1]["content"] = messages[-1]["content"][:len(messages[-1]["content"]) - excess]
            else:
                while total_chars > self.LM_MAX_TOTAL_CHARS and len(messages) > 2:
                    removed = messages.pop(1)
                    self.logger.warning(f"Removed old history message to fit LM payload: {removed}")
                    total_chars = sum(len(m["content"]) for m in messages)
                if total_chars > self.LM_MAX_TOTAL_CHARS:
                    last = messages[-1]
                    last["content"] = last["content"][:max(0, len(last["content"]) - (total_chars - self.LM_MAX_TOTAL_CHARS))]
        return messages

    def _make_request(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k

        self.logger.debug(f"FreeGPT4Client: Sending chat payload to {self.url}: {str(payload)[:800]}")

        try:
            response = requests.post(self.url, json=payload, timeout=self.timeout)
        except Exception as e:
            self.logger.error(f"Error during POST to FreeGPT4 API: {e}")
            return ""
        if not response.ok:
            self.logger.error(f"FreeGPT4 API response HTTP error: {response.status_code} {response.text[:200]}")
            return ""
        self.logger.debug(f"FreeGPT4Client: raw response: {response.text[:1000]}")
        try:
            result = response.json()
        except Exception as e:
            self.logger.error("Failed to decode FreeGPT4 API response as JSON", exc_info=True)
            result = {}

        text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not isinstance(text, str):
            raise ValueError("FreeGPT4 API returned non-string result.")
        return text.strip()

    def _save_lm_log(self, text: str, topic: str, success: bool, prompt: Optional[str] = None, attempt: int = 0):
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
        folder = self.log_dir_success if success else self.log_dir_failed
        filename = f"{date_str}_attempt{attempt}_{safe_topic[:40]}.txt"
        try:
            with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            self.logger.error(f"Failed to save LM log: {e}")
        if prompt:
            try:
                with open(os.path.join(self.log_dir_prompts, f"{date_str}_attempt{attempt}_{safe_topic[:40]}_prompt.txt"), "w", encoding="utf-8") as f:
                    f.write(prompt)
            except Exception as e:
                self.logger.error(f"Failed to save LM prompt log: {e}")

    def generate_content(self, prompt_template: str, topic: str, context: str, max_tokens: Optional[int] = None) -> str:
        """
        Генерирует текст с учетом лимитов Telegram, с автоматическим сокращением и логированием.
        Контент дополнительно требует пост-валидации на уровне вызывающего кода.
        """
        max_tokens = max_tokens or self.max_tokens
        original_max_tokens = self.max_tokens
        self.max_tokens = max_tokens
        text = ""
        last_prompt = ""
        clean_text = ""
        try:
            messages = self._build_messages(prompt_template, topic, context)
            last_prompt = messages[-1]['content'] if messages else ""
            text = self._make_request(messages)
            # ВАЖНО: очищаем от размышлений сразу
            clean_text = self.content_validator.validate_content(text)
            self._save_lm_log(clean_text, topic, False, last_prompt, attempt=0)
            if not clean_text or not clean_text.strip():
                self.logger.warning("FreeGPT4 API returned empty/invalid text after validation.")
                return ""

            # Проверяем длину только очищенного текста!
            if len(clean_text) > self.TELEGRAM_LIMIT:
                self.logger.warning(f"Generated content too long ({len(clean_text)} chars), requesting shorter version")
                attempts = 0
                max_attempts = 3
                context_for_shorter = context
                while len(clean_text) > self.TELEGRAM_LIMIT and attempts < max_attempts:
                    attempts += 1
                    self.clear_conversation_history()
                    shorter_prompt = (
                        f"{prompt_template}\n\nВАЖНО: Предыдущий ответ был слишком длинный ({len(clean_text)} символов). "
                        "Сократи, убери детали, не добавляй размышления или пояснения."
                    )
                    messages = self._build_messages(shorter_prompt, topic, context_for_shorter)
                    last_prompt = messages[-1]['content'] if messages else ""
                    text = self._make_request(messages)
                    clean_text = self.content_validator.validate_content(text)
                    self._save_lm_log(clean_text, topic, False, last_prompt, attempt=attempts)
                    if not clean_text or not clean_text.strip():
                        self.logger.warning("Shorter version attempt returned empty text.")
                        break
                if len(clean_text) > self.TELEGRAM_LIMIT:
                    self.logger.warning("Unable to get short enough text. Truncating and cleaning again.")
                    # Обрезаем, затем еще раз валидируем (на случай разорванных <think>)
                    clean_text = self.content_validator.validate_content(clean_text[:self.TELEGRAM_LIMIT-10] + "...")

            # Логируем только очищенный текст, который реально будет отправлен
            if clean_text and clean_text.strip() and len(clean_text) <= self.TELEGRAM_LIMIT:
                self._save_lm_log(clean_text, topic, True, last_prompt, attempt=99)
                self.add_to_history(last_prompt, clean_text)
            else:
                self.logger.warning("Generated content is too long or empty after all shortening attempts.")

            return clean_text.strip() if clean_text else ""
        finally:
            self.max_tokens = original_max_tokens

    def generate_with_retry(self, prompt_template: str, topic: str, context: str, max_retries: int = 3) -> str:
        """
        Генерирует текст с автоматическими повторными попытками при ошибках.
        - При HTTP 413/400 или payload error уменьшает контекст.
        - При повторных ошибках очищает историю.
        - Последняя попытка с минимальным контекстом.
        """
        last_err = None
        original_context = context
        final_text = ""
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.debug(f"FreeGPT4Client: generation attempt {attempt}/{max_retries}")
                text = self.generate_content(prompt_template, topic, context)
                if text and text.strip() and len(text) <= self.TELEGRAM_LIMIT:
                    final_text = text
                    break
                self.logger.warning(f"FreeGPT4 API generation returned empty or too long text (attempt {attempt})")
            except Exception as e:
                last_err = e
                msg = str(e)
                self.logger.warning(f"FreeGPT4Client: error on attempt {attempt}: {msg}")
                if "413" in msg or "400" in msg or "payload" in msg:
                    context = context[:max(100, len(context) // 2)]
                    self.logger.warning("Reducing context and retrying...")
                if attempt == max_retries or (self.history_limit and len(self.history) > self.history_limit * 4):
                    self.logger.warning("Clearing conversation history due to repeated failures.")
                    self.clear_conversation_history()
        # Если ни одна попытка не прошла, делаем финальный fallback
        if not final_text:
            try:
                text = self.generate_content(prompt_template, topic, original_context[:256])
                if text and text.strip() and len(text) <= self.TELEGRAM_LIMIT:
                    final_text = text
            except Exception as e:
                self.logger.error("Final fallback attempt failed", exc_info=True)
                raise ValueError(f"FreeGPT4 API did not generate content after {max_retries} attempts: {last_err}")
        return final_text.strip() if final_text else ""
