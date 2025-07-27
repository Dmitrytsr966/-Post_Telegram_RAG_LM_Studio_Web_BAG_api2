import logging
import requests
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from modules.content_generation.content_validator import ContentValidator

class CircuitBreaker:
    """Circuit breaker для управления паузами при критических сбоях всех провайдеров"""
    
    def __init__(self, failure_threshold: int = 3, pause_duration: int = 30, reset_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.pause_duration = pause_duration
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = logging.getLogger("CircuitBreaker")
    
    def record_success(self):
        """Записать успешный запрос"""
        self.failure_count = 0
        self.state = "CLOSED"
        self.logger.debug("Circuit breaker: success recorded, state reset to CLOSED")
    
    def record_failure(self):
        """Записать неудачный запрос"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures. Pause for {self.pause_duration}s")
    
    def can_attempt(self) -> bool:
        """Можно ли делать запрос"""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            if self.last_failure_time and (datetime.now() - self.last_failure_time).total_seconds() > self.pause_duration:
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker changed to HALF_OPEN, allowing test request")
                return True
            return False
        
        if self.state == "HALF_OPEN":
            return True
        
        return False
    
    def should_pause(self) -> int:
        """Вернуть количество секунд паузы, если нужна пауза"""
        if self.state == "OPEN" and self.last_failure_time:
            elapsed = (datetime.now() - self.last_failure_time).total_seconds()
            remaining = max(0, self.pause_duration - elapsed)
            return int(remaining)
        return 0

class EndpointManager:
    """Менеджер для управления множественными endpoints и их ротацией"""
    
    def __init__(self, primary_endpoint: Dict, fallback_endpoints: List[Dict], logger):
        self.primary_endpoint = primary_endpoint
        self.fallback_endpoints = fallback_endpoints
        self.all_endpoints = [primary_endpoint] + fallback_endpoints
        self.current_index = 0
        self.logger = logger
        self.failed_endpoints = set()  # Временно недоступные endpoints
        self.failure_timestamps = {}  # Время последней ошибки для каждого endpoint
        self.failure_reset_time = 300  # 5 минут до сброса статуса ошибки
        
        self.logger.info(f"EndpointManager initialized with {len(self.all_endpoints)} endpoints")
        for i, endpoint in enumerate(self.all_endpoints):
            self.logger.info(f"  Endpoint {i}: {endpoint['provider_name']} - {endpoint['url']}")
    
    def get_current_endpoint(self) -> Dict:
        """Получить текущий активный endpoint"""
        self._cleanup_old_failures()
        
        # Попробовать найти работающий endpoint начиная с текущего
        for _ in range(len(self.all_endpoints)):
            endpoint = self.all_endpoints[self.current_index]
            endpoint_id = self._get_endpoint_id(endpoint)
            
            if endpoint_id not in self.failed_endpoints:
                return endpoint
            
            self.current_index = (self.current_index + 1) % len(self.all_endpoints)
        
        # Если все endpoints в failed состоянии, вернуть текущий (возможно, они восстановились)
        self.logger.warning("All endpoints are marked as failed, returning current anyway")
        return self.all_endpoints[self.current_index]
    
    def mark_endpoint_failed(self, endpoint: Dict, error: str):
        """Отметить endpoint как неработающий"""
        endpoint_id = self._get_endpoint_id(endpoint)
        self.failed_endpoints.add(endpoint_id)
        self.failure_timestamps[endpoint_id] = datetime.now()
        
        self.logger.warning(f"Endpoint {endpoint['provider_name']} marked as failed: {error}")
        
        # Переключиться на следующий endpoint
        self.current_index = (self.current_index + 1) % len(self.all_endpoints)
    
    def mark_endpoint_success(self, endpoint: Dict):
        """Отметить endpoint как работающий"""
        endpoint_id = self._get_endpoint_id(endpoint)
        if endpoint_id in self.failed_endpoints:
            self.failed_endpoints.remove(endpoint_id)
            self.failure_timestamps.pop(endpoint_id, None)
            self.logger.info(f"Endpoint {endpoint['provider_name']} restored to working state")
    
    def get_next_endpoint(self) -> Optional[Dict]:
        """Получить следующий доступный endpoint"""
        self._cleanup_old_failures()
        
        original_index = self.current_index
        self.current_index = (self.current_index + 1) % len(self.all_endpoints)
        
        # Найти следующий работающий endpoint
        for _ in range(len(self.all_endpoints) - 1):
            endpoint = self.all_endpoints[self.current_index]
            endpoint_id = self._get_endpoint_id(endpoint)
            
            if endpoint_id not in self.failed_endpoints:
                self.logger.info(f"Switched to endpoint: {endpoint['provider_name']}")
                return endpoint
            
            self.current_index = (self.current_index + 1) % len(self.all_endpoints)
        
        # Если нет доступных endpoints, вернуть None
        self.current_index = original_index
        return None
    
    def _get_endpoint_id(self, endpoint: Dict) -> str:
        """Получить уникальный ID endpoint"""
        return f"{endpoint['provider_name']}_{endpoint['url']}"
    
    def _cleanup_old_failures(self):
        """Очистить старые записи об ошибках"""
        current_time = datetime.now()
        expired_endpoints = []
        
        for endpoint_id, failure_time in self.failure_timestamps.items():
            if (current_time - failure_time).total_seconds() > self.failure_reset_time:
                expired_endpoints.append(endpoint_id)
        
        for endpoint_id in expired_endpoints:
            self.failed_endpoints.discard(endpoint_id)
            self.failure_timestamps.pop(endpoint_id, None)
            self.logger.info(f"Endpoint failure status expired and reset: {endpoint_id}")
    
    def get_stats(self) -> Dict:
        """Получить статистику endpoints"""
        return {
            "total_endpoints": len(self.all_endpoints),
            "current_endpoint": self.all_endpoints[self.current_index]['provider_name'],
            "failed_endpoints": len(self.failed_endpoints),
            "available_endpoints": len(self.all_endpoints) - len(self.failed_endpoints)
        }

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
        + Добавлена поддержка множественных endpoints для ротации провайдеров
        """
        self.logger = logging.getLogger("FreeGPT4Client")
        
        # НОВОЕ: Инициализация endpoint manager для ротации провайдеров
        primary_endpoint = config.get("primary_endpoint", {
            "url": url,
            "provider_name": "Default",
            "model": model,
            "timeout": 60
        })
        
        fallback_endpoints = config.get("fallback_endpoints", [])
        
        self.endpoint_manager = EndpointManager(primary_endpoint, fallback_endpoints, self.logger)
        
        # ОРИГИНАЛЬНЫЕ параметры из исходного кода - НЕ ИЗМЕНЕНЫ
        self.url = url.rstrip("/")  # Сохраняем для обратной совместимости
        self.model = model
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.7)
        self.timeout = config.get("timeout", 60)
        # Устанавливаем history_limit = 1 для предотвращения накопления ошибок
        self.history_limit = 0
        self.system_message = config.get("system_message", None)
        self.top_p = config.get("top_p", None)
        self.top_k = config.get("top_k", None)
        
        # НОВЫЕ настройки ротации провайдеров
        self.switch_errors = config.get("provider_switch_on_errors", ["400", "429", "503", "timeout", "connection_error"])
        self.max_provider_retries = config.get("max_provider_retries", 2)
        
        # НОВЫЙ Circuit breaker
        cb_config = config.get("circuit_breaker", {})
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=cb_config.get("failure_threshold", 3),
            pause_duration=cb_config.get("pause_duration_seconds", 30),
            reset_timeout=cb_config.get("reset_timeout_seconds", 300)
        ) if cb_config.get("enabled", True) else None
        
        # ОРИГИНАЛЬНЫЕ переменные из исходного кода - НЕ ИЗМЕНЕНЫ
        self.history: List[Dict[str, str]] = []
        self.content_validator = ContentValidator(config=config)
        self.log_dir_success = "logs/freegpt4/success"
        self.log_dir_failed = "logs/freegpt4/failed"
        self.log_dir_prompts = "logs/freegpt4/prompts"
        os.makedirs(self.log_dir_success, exist_ok=True)
        os.makedirs(self.log_dir_failed, exist_ok=True)
        os.makedirs(self.log_dir_prompts, exist_ok=True)
        
        self._validate_config()
        
        # НОВОЕ: Лог статистики endpoints
        stats = self.endpoint_manager.get_stats()
        self.logger.info(f"FreeGPT4Client initialized with endpoint rotation: {stats}")

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
        # С history_limit = 0, не ограничиваем историю (или можно настроить)
        if self.history_limit > 0 and len(self.history) > self.history_limit * 2:
            self.history = self.history[-self.history_limit * 2:]

    def _clean_history(self) -> List[Dict[str, str]]:
        clean = []
        for m in self.history[-self.history_limit * 2:] if self.history_limit > 0 else self.history:
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
        """Рассчитывает доступное место для контекста с учетом всех статических элементов"""
        system_size = len(self.system_message) if self.system_message else 0
        history_size = sum(len(m["content"]) for m in self._clean_history())
        prompt_without_context = prompt_template.replace("{TOPIC}", topic.strip()).replace("{CONTEXT}", "")
        prompt_size = len(prompt_without_context)
        static_size = system_size + history_size + prompt_size + self.RESERVE_CHARS
        available = self.LM_MAX_TOTAL_CHARS - static_size
        
        self.logger.debug(f"Context space calculation: system={system_size}, history={history_size}, prompt={prompt_size}, reserve={self.RESERVE_CHARS}, available={available}")
        
        return max(0, available)

    def _truncate_context_for_llm(self, prompt_template: str, topic: str, context: str) -> str:
        """Обрезает контекст с учетом реального доступного места"""
        available = self._calculate_available_context_space(prompt_template, topic)
        
        if available <= 100:
            self.logger.warning(f"Very little space for context: {available} chars. Clearing history.")
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
            sentences = context.split('.')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + ".") <= available:
                    truncated += sentence + "."
                else:
                    break
            
            if not truncated:
                truncated = context[:available]
            
            context = truncated
        
        return context

    def _build_messages(self, prompt_template: str, topic: str, context: str) -> List[Dict[str, str]]:
        """Строит сообщения с улучшенным контролем размера"""
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
            
            while total_chars > self.LM_MAX_TOTAL_CHARS and len(messages) > 2:
                if len(messages) > 2:
                    removed = messages.pop(1)
                    self.logger.warning(f"Removed old history message: {removed['role']}")
                    total_chars = sum(len(m["content"]) for m in messages)
            
            if total_chars > self.LM_MAX_TOTAL_CHARS:
                excess = total_chars - self.LM_MAX_TOTAL_CHARS
                current_prompt = messages[-1]["content"]
                if len(current_prompt) > excess:
                    messages[-1]["content"] = current_prompt[:len(current_prompt) - excess - 10] + "..."
                    self.logger.warning(f"Trimmed current prompt by {excess} chars")
        
        final_total = sum(len(m["content"]) for m in messages)
        self.logger.debug(f"Final message payload size: {final_total} chars")
        
        return messages

    def _should_switch_provider(self, error_msg: str, status_code: Optional[int] = None) -> bool:
        """Определяет, нужно ли переключиться на другой провайдер"""
        error_msg_lower = error_msg.lower()
        
        # Проверка по коду ошибки
        if status_code:
            if str(status_code) in self.switch_errors:
                return True
        
        # Проверка по тексту ошибки
        for error_pattern in self.switch_errors:
            if error_pattern.lower() in error_msg_lower:
                return True
        
        # Специфичные ошибки, требующие смены провайдера
        switch_patterns = [
            "not enough credits",
            "rate limit",
            "timeout",
            "connection error",
            "503",
            "502",
            "500",
            "429"
        ]
        
        for pattern in switch_patterns:
            if pattern in error_msg_lower:
                return True
        
        return False

    def _make_request(self, messages: List[Dict[str, str]], endpoint: Optional[Dict] = None) -> str:
        """
        ОБНОВЛЕННАЯ версия оригинального метода _make_request с поддержкой endpoint ротации
        Если endpoint не указан, использует оригинальную логику с self.url
        """
        if endpoint is None:
            # ОРИГИНАЛЬНОЕ поведение для обратной совместимости
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
        
        else:
            # НОВОЕ поведение с конкретным endpoint
            payload = {
                "model": endpoint.get("model", "gpt-4"),
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            if self.top_p is not None:
                payload["top_p"] = self.top_p
            if self.top_k is not None:
                payload["top_k"] = self.top_k

            url = endpoint["url"]
            timeout = endpoint.get("timeout", self.timeout)
            provider_name = endpoint.get("provider_name", "Unknown")

            self.logger.debug(f"Making request to {provider_name} ({url}): {str(payload)[:800]}")

            try:
                response = requests.post(url, json=payload, timeout=timeout)
            except requests.exceptions.Timeout as e:
                raise Exception(f"Timeout error for {provider_name}: {str(e)}")
            except requests.exceptions.ConnectionError as e:
                raise Exception(f"Connection error for {provider_name}: {str(e)}")
            except Exception as e:
                raise Exception(f"Request error for {provider_name}: {str(e)}")

            if not response.ok:
                error_msg = f"HTTP {response.status_code} from {provider_name}: {response.text[:200]}"
                self.logger.error(error_msg)
                raise Exception(error_msg)

            self.logger.debug(f"{provider_name} response: {response.text[:1000]}")
            
            try:
                result = response.json()
            except Exception as e:
                raise Exception(f"Failed to decode JSON from {provider_name}: {str(e)}")

            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not isinstance(text, str):
                raise Exception(f"{provider_name} returned non-string result")
            
            return text.strip()

    def _save_lm_log(self, text: str, topic: str, success: bool, prompt: Optional[str] = None, attempt: int = 0, provider: str = "unknown"):
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
        folder = self.log_dir_success if success else self.log_dir_failed
        filename = f"{date_str}_attempt{attempt}_{provider}_{safe_topic[:40]}.txt"
        try:
            with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            self.logger.error(f"Failed to save LM log: {e}")
        if prompt:
            try:
                with open(os.path.join(self.log_dir_prompts, f"{date_str}_attempt{attempt}_{provider}_{safe_topic[:40]}_prompt.txt"), "w", encoding="utf-8") as f:
                    f.write(prompt)
            except Exception as e:
                self.logger.error(f"Failed to save LM prompt log: {e}")

    def generate_content(self, prompt_template: str, topic: str, context: str, max_tokens: Optional[int] = None) -> str:
        """
        РАСШИРЕННАЯ версия оригинального метода generate_content.
        Генерирует текст с учетом лимитов Telegram, с дублированием запроса при превышении лимита.
        + Добавлена поддержка ротации провайдеров при ошибках.
        Контент дополнительно требует пост-валидации на уровне вызывающего кода.
        """
        max_tokens = max_tokens or self.max_tokens
        original_max_tokens = self.max_tokens
        self.max_tokens = max_tokens
        text = ""
        last_prompt = ""
        clean_text = ""
        
        try:
            # Основной запрос с поддержкой ротации провайдеров
            messages = self._build_messages(prompt_template, topic, context)
            last_prompt = messages[-1]['content'] if messages else ""
            
            # НОВОЕ: Попробовать текущий endpoint из менеджера
            current_endpoint = self.endpoint_manager.get_current_endpoint()
            
            try:
                text = self._make_request(messages, current_endpoint)
                
                # ОРИГИНАЛЬНАЯ логика валидации контента - НЕ ИЗМЕНЕНА
                clean_text = self.content_validator.validate_content(text)
                self._save_lm_log(clean_text, topic, False, last_prompt, attempt=0, 
                                provider=current_endpoint["provider_name"])
                
                if not clean_text or not clean_text.strip():
                    self.logger.warning("FreeGPT4 API returned empty/invalid text after validation.")
                    # НОВОЕ: При пустом ответе попробовать fallback
                    next_endpoint = self.endpoint_manager.get_next_endpoint()
                    if next_endpoint:
                        self.logger.info(f"Trying fallback {next_endpoint['provider_name']} due to empty response")
                        text = self._make_request(messages, next_endpoint)
                        clean_text = self.content_validator.validate_content(text)
                        self._save_lm_log(clean_text, topic, False, last_prompt, attempt=1,
                                        provider=next_endpoint["provider_name"])
                    
                    if not clean_text or not clean_text.strip():
                        return ""

                # ОРИГИНАЛЬНАЯ логика проверки длины и дублирования запросов - НЕ ИЗМЕНЕНА
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
                        
                        # НОВОЕ: Используем тот же endpoint или fallback при ошибке
                        try:
                            duplicate_text = self._make_request(duplicate_messages, current_endpoint)
                        except Exception as e:
                            if self._should_switch_provider(str(e)):
                                next_endpoint = self.endpoint_manager.get_next_endpoint()
                                if next_endpoint:
                                    duplicate_text = self._make_request(duplicate_messages, next_endpoint)
                                    current_endpoint = next_endpoint
                                else:
                                    raise e
                            else:
                                raise e
                        
                        # Валидируем дублированный ответ
                        duplicate_clean_text = self.content_validator.validate_content(duplicate_text)
                        self._save_lm_log(duplicate_clean_text, topic, False, last_prompt, attempt=attempt,
                                        provider=current_endpoint["provider_name"])
                        
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

                # ОРИГИНАЛЬНОЕ логирование успешного результата - НЕ ИЗМЕНЕНО
                if clean_text and clean_text.strip() and len(clean_text) <= self.TELEGRAM_LIMIT:
                    self._save_lm_log(clean_text, topic, True, last_prompt, attempt=99,
                                    provider=current_endpoint["provider_name"])
                    # НЕ добавляем в историю если это был дублированный запрос, чтобы избежать накопления
                    # Добавляем только успешный финальный результат
                    self.add_to_history(last_prompt, clean_text)
                    
                    # НОВОЕ: Отметить endpoint как работающий
                    self.endpoint_manager.mark_endpoint_success(current_endpoint)
                    if self.circuit_breaker:
                        self.circuit_breaker.record_success()
                else:
                    self.logger.warning("Generated content is too long or empty after all duplicate attempts.")

                return clean_text.strip() if clean_text else ""
                
            except Exception as e:
                error_msg = str(e)
                self.logger.warning(f"Error from {current_endpoint['provider_name']}: {error_msg}")
                
                # НОВОЕ: Определить, нужно ли переключаться на другой провайдер
                if self._should_switch_provider(error_msg):
                    self.endpoint_manager.mark_endpoint_failed(current_endpoint, error_msg)
                    
                    # Попробовать следующий endpoint
                    next_endpoint = self.endpoint_manager.get_next_endpoint()
                    if next_endpoint:
                        self.logger.info(f"Switching to {next_endpoint['provider_name']} due to error")
                        try:
                            text = self._make_request(messages, next_endpoint)
                            clean_text = self.content_validator.validate_content(text)
                            
                            if clean_text and clean_text.strip():
                                # Успешный запрос с fallback endpoint
                                self.endpoint_manager.mark_endpoint_success(next_endpoint)
                                if self.circuit_breaker:
                                    self.circuit_breaker.record_success()
                                
                                self._save_lm_log(clean_text, topic, True, last_prompt, attempt=1, 
                                                provider=next_endpoint["provider_name"])
                                self.add_to_history(last_prompt, clean_text)
                                return clean_text
                            else:
                                self.logger.warning(f"Empty content from fallback {next_endpoint['provider_name']}")
                                
                        except Exception as fallback_error:
                            self.logger.error(f"Fallback {next_endpoint['provider_name']} also failed: {str(fallback_error)}")
                            self.endpoint_manager.mark_endpoint_failed(next_endpoint, str(fallback_error))
                
                # Если все endpoints недоступны, записать ошибку в circuit breaker
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()
                
                raise Exception(f"All providers failed. Last error: {error_msg}")
            
        finally:
            self.max_tokens = original_max_tokens

    def generate_with_retry(self, prompt_template: str, topic: str, context: str, max_retries: int = 3) -> str:
        """Генерирует текст с автоматическими повторными попытками и circuit breaker"""
        
        # Проверка circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.can_attempt():
            pause_time = self.circuit_breaker.should_pause()
            if pause_time > 0:
                self.logger.warning(f"Circuit breaker is OPEN. Pausing for {pause_time} seconds...")
                time.sleep(pause_time)
                
                # После паузы проверить снова
                if not self.circuit_breaker.can_attempt():
                    raise Exception("Circuit breaker is OPEN - all providers unavailable")
        
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
                    reduction_factor = 0.5 ** attempt
                    context = context[:max(200, int(len(context) * reduction_factor))]
                    self.logger.warning(f"Reducing context to {len(context)} chars and clearing history...")
                    self.clear_conversation_history()
                
                # При timeout или частых ошибках
                if attempt >= 2:
                    self.logger.warning("Multiple failures, clearing conversation history.")
                    self.clear_conversation_history()
                
                # Проверка circuit breaker после каждой ошибки
                if self.circuit_breaker and not self.circuit_breaker.can_attempt():
                    pause_time = self.circuit_breaker.should_pause()
                    if pause_time > 0:
                        self.logger.warning(f"Circuit breaker activated, pausing for {pause_time} seconds...")
                        time.sleep(pause_time)
        
        # Если ни одна попытка не прошла, делаем финальный fallback с минимальным контекстом
        if not final_text:
            try:
                self.logger.warning("Final fallback attempt with minimal context")
                
                # Проверка circuit breaker перед финальной попыткой
                if self.circuit_breaker and not self.circuit_breaker.can_attempt():
                    raise Exception("Circuit breaker prevents final attempt")
                
                self.clear_conversation_history()
                minimal_context = original_context[:400]
                text = self.generate_content(prompt_template, topic, minimal_context)
                if text and text.strip() and len(text) <= self.TELEGRAM_LIMIT:
                    final_text = text
            except Exception as e:
                self.logger.error("Final fallback attempt failed", exc_info=True)
                raise ValueError(f"FreeGPT4 API did not generate content after {max_retries} attempts: {last_err}")
        
        return final_text.strip() if final_text else ""
