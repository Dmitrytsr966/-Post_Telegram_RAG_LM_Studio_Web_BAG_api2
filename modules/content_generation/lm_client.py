import logging
import requests
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from modules.content_generation.content_validator import ContentValidator

class FreeGPT4Client:
    LM_MAX_TOTAL_CHARS = 4500
    TELEGRAM_LIMIT = 4096
    # Резерв для служебных символов JSON и форматирования
    RESERVE_CHARS = 500

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
        # Устанавливаем history_limit = 1 для предотвращения накопления ошибок
        self.history_limit = 0
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
        # С history_limit = 1, оставляем только последнюю пару сообщений
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

    def _calculate_available_context_space(self, prompt_template: str, topic: str) -> int:
        """
        Рассчитывает доступное место для контекста с учетом всех статических элементов
        """
        # Расчет размера системного сообщения
        system_size = len(self.system_message) if self.system_message else 0
        
        # Расчет размера истории
        history_size = sum(len(m["content"]) for m in self._clean_history())
        
        # Расчет размера промпта без контекста
        prompt_without_context = prompt_template.replace("{TOPIC}", topic.strip()).replace("{CONTEXT}", "")
        prompt_size = len(prompt_without_context)
        
        # Общий размер статических элементов
        static_size = system_size + history_size + prompt_size + self.RESERVE_CHARS
        
        # Доступное место для контекста
        available = self.LM_MAX_TOTAL_CHARS - static_size
        
        self.logger.debug(f"Context space calculation: system={system_size}, history={history_size}, prompt={prompt_size}, reserve={self.RESERVE_CHARS}, available={available}")
        
        return max(0, available)

    def _truncate_context_for_llm(self, prompt_template: str, topic: str, context: str) -> str:
        """
        Обрезает контекст с учетом реального доступного места
        """
        available = self._calculate_available_context_space(prompt_template, topic)
        
        if available <= 100:  # Минимальный размер контекста
            self.logger.warning(f"Very little space for context: {available} chars. Clearing history.")
            # Очищаем историю и пересчитываем
            old_history = self.history.copy()
            self.clear_conversation_history()
            available = self._calculate_available_context_space(prompt_template, topic)
            
            if available <= 100:
                self.logger.error(f"Still no room for context after clearing history: {available} chars")
                return context[:100] if context else ""
            else:
                self.logger.info(f"After clearing history, available space: {available} chars")
        
        context = context.strip()
        if len(context) > available:
            self.logger.warning(f"Context too long for LLM ({len(context)} > {available}), truncating context")
            # Обрезаем по предложениям, чтобы не разрывать смысл
            sentences = context.split('.')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + ".") <= available:
                    truncated += sentence + "."
                else:
                    break
            
            if not truncated:  # Если даже одно предложение не помещается
                truncated = context[:available]
            
            context = truncated
        
        return context

    def _build_messages(self, prompt_template: str, topic: str, context: str) -> List[Dict[str, str]]:
        """
        Строит сообщения с улучшенным контролем размера
        """
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

        # Финальная проверка общего размера
        total_chars = sum(len(m["content"]) for m in messages)
        
        if total_chars > self.LM_MAX_TOTAL_CHARS:
            self.logger.warning(f"Total LLM payload still too long ({total_chars} > {self.LM_MAX_TOTAL_CHARS}), final trimming")
            
            # Сначала пытаемся удалить историю
            while total_chars > self.LM_MAX_TOTAL_CHARS and len(messages) > 2:
                # Удаляем самое старое сообщение из истории (после системного)
                if len(messages) > 2:
                    removed = messages.pop(1)
                    self.logger.warning(f"Removed old history message: {removed['role']}")
                    total_chars = sum(len(m["content"]) for m in messages)
            
            # Если все еще слишком длинно, обрезаем текущий промпт
            if total_chars > self.LM_MAX_TOTAL_CHARS:
                excess = total_chars - self.LM_MAX_TOTAL_CHARS
                current_prompt = messages[-1]["content"]
                if len(current_prompt) > excess:
                    messages[-1]["content"] = current_prompt[:len(current_prompt) - excess - 10] + "..."
                    self.logger.warning(f"Trimmed current prompt by {excess} chars")
        
        final_total = sum(len(m["content"]) for m in messages)
        self.logger.debug(f"Final message payload size: {final_total} chars")
        
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
        Генерирует текст с учетом лимитов Telegram, с дублированием запроса при превышении лимита.
        Контент дополнительно требует пост-валидации на уровне вызывающего кода.
        """
        max_tokens = max_tokens or self.max_tokens
        original_max_tokens = self.max_tokens
        self.max_tokens = max_tokens
        text = ""
        last_prompt = ""
        clean_text = ""
        
        try:
            # Основной запрос
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
                self.logger.warning(f"Generated content too long ({len(clean_text)} chars), making duplicate request")
                
                # ИСПРАВЛЕНИЕ: вместо модификации промпта делаем дублирующий запрос
                max_attempts = 3
                for attempt in range(1, max_attempts + 1):
                    self.logger.info(f"Duplicate request attempt {attempt}/{max_attempts}")
                    
                    # Очищаем историю перед каждым дублированием
                    self.clear_conversation_history()
                    
                    # Делаем точно такой же запрос без модификации промпта
                    duplicate_messages = self._build_messages(prompt_template, topic, context)
                    last_prompt = duplicate_messages[-1]['content'] if duplicate_messages else ""
                    duplicate_text = self._make_request(duplicate_messages)
                    
                    # Валидируем дублированный ответ
                    duplicate_clean_text = self.content_validator.validate_content(duplicate_text)
                    self._save_lm_log(duplicate_clean_text, topic, False, last_prompt, attempt=attempt)
                    
                    if not duplicate_clean_text or not duplicate_clean_text.strip():
                        self.logger.warning(f"Duplicate attempt {attempt} returned empty text.")
                        continue
                    
                    # Если дублированный запрос дал нужную длину - используем его
                    if len(duplicate_clean_text) <= self.TELEGRAM_LIMIT:
                        clean_text = duplicate_clean_text
                        self.logger.info(f"Duplicate request {attempt} successful, length: {len(clean_text)}")
                        break
                    else:
                        self.logger.warning(f"Duplicate attempt {attempt} still too long: {len(duplicate_clean_text)} chars")
                
                # Если все дубликаты были слишком длинными, обрезаем последний результат
                if len(clean_text) > self.TELEGRAM_LIMIT:
                    self.logger.warning("All duplicate requests were too long. Truncating and cleaning again.")
                    # Обрезаем, затем еще раз валидируем (на случай разорванных <think>)
                    clean_text = self.content_validator.validate_content(clean_text[:self.TELEGRAM_LIMIT-10] + "...")

            # Логируем только очищенный текст, который реально будет отправлен
            if clean_text and clean_text.strip() and len(clean_text) <= self.TELEGRAM_LIMIT:
                self._save_lm_log(clean_text, topic, True, last_prompt, attempt=99)
                # НЕ добавляем в историю если это был дублированный запрос, чтобы избежать накопления
                # Добавляем только успешный финальный результат
                self.add_to_history(last_prompt, clean_text)
            else:
                self.logger.warning("Generated content is too long or empty after all duplicate attempts.")

            return clean_text.strip() if clean_text else ""
            
        finally:
            self.max_tokens = original_max_tokens

    def generate_with_retry(self, prompt_template: str, topic: str, context: str, max_retries: int = 3) -> str:
        """
        Генерирует текст с автоматическими повторными попытками при ошибках.
        - При HTTP 413/400 или payload error уменьшает контекст и очищает историю.
        - При повторных ошибках агрессивно очищает историю.
        - Последняя попытка с минимальным контекстом.
        """
        last_err = None
        original_context = context
        final_text = ""
        
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.debug(f"FreeGPT4Client: generation attempt {attempt}/{max_retries}")
                
                # При повторных попытках агрессивно очищаем историю
                if attempt > 1:
                    self.logger.warning(f"Clearing history before attempt {attempt}")
                    self.clear_conversation_history()
                
                text = self.generate_content(prompt_template, topic, context)
                if text and text.strip() and len(text) <= self.TELEGRAM_LIMIT:
                    final_text = text
                    break
                self.logger.warning(f"FreeGPT4 API generation returned empty or too long text (attempt {attempt})")
                
            except Exception as e:
                last_err = e
                msg = str(e)
                self.logger.warning(f"FreeGPT4Client: error on attempt {attempt}: {msg}")
                
                # При ошибках связанных с размером payload
                if "413" in msg or "400" in msg or "payload" in msg or "timeout" in msg:
                    # Уменьшаем контекст более агрессивно
                    reduction_factor = 0.5 ** attempt  # Каждая попытка уменьшает контекст в 2 раза
                    context = context[:max(200, int(len(context) * reduction_factor))]
                    self.logger.warning(f"Reducing context to {len(context)} chars and clearing history...")
                    self.clear_conversation_history()
                
                # При timeout или частых ошибках
                if attempt >= 2:
                    self.logger.warning("Multiple failures, clearing conversation history.")
                    self.clear_conversation_history()
        
        # Если ни одна попытка не прошла, делаем финальный fallback с минимальным контекстом
        if not final_text:
            try:
                self.logger.warning("Final fallback attempt with minimal context")
                self.clear_conversation_history()
                minimal_context = original_context[:400]  # Минимальный контекст
                text = self.generate_content(prompt_template, topic, minimal_context)
                if text and text.strip() and len(text) <= self.TELEGRAM_LIMIT:
                    final_text = text
            except Exception as e:
                self.logger.error("Final fallback attempt failed", exc_info=True)
                raise ValueError(f"FreeGPT4 API did not generate content after {max_retries} attempts: {last_err}")
        
        return final_text.strip() if final_text else ""
