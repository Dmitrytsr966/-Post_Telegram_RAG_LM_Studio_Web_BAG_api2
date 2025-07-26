# modules/external_apis/web_search.py

import logging
import requests
import time
import hashlib
from typing import List, Dict
from pathlib import Path
from datetime import datetime, timedelta

class WebSearchClient:
    def __init__(self, api_key: str, endpoint: str = "https://google.serper.dev/search", results_limit: int = 10, cache_ttl_minutes: int = 60):
        self.api_key = api_key
        self.endpoint = endpoint
        self.results_limit = results_limit
        self.logger = logging.getLogger("WebSearchClient")
        self.circuit_breaker_tripped = False  # Флаг для активации заглушки
        
        # Кэширование
        self.cache = {}
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)

    def _get_cache_key(self, query: str, num_results: int) -> str:
        """Создает ключ кэша для запроса"""
        content = f"{query}:{num_results}"
        return hashlib.md5(content.encode()).hexdigest()

    def _preprocess_query(self, query: str) -> str:
        """Предобработка поискового запроса"""
        # Удаление лишних символов
        query = query.strip()
        
        # Замена специальных символов
        query = query.replace('"', '').replace("'", "")
        
        # Ограничение длины
        if len(query) > 200:
            query = query[:200]
        
        return query

    def _filter_quality_results(self, results: List[Dict], min_snippet_length: int = 50) -> List[Dict]:
        """Фильтрация результатов по качеству"""
        filtered = []
        for result in results:
            snippet = result.get("snippet", "")
            title = result.get("title", "")
            
            # Пропускаем результаты с короткими сниппетами
            if len(snippet) < min_snippet_length:
                continue
                
            # Пропускаем результаты без заголовка
            if not title:
                continue
                
            # Проверяем на спам-индикаторы
            spam_indicators = ["порно", "новости", "арест", "эскорт", "интим", "начать играть"]
            if any(indicator in title.lower() or indicator in snippet.lower() for indicator in spam_indicators):
                continue
                
            filtered.append(result)
        
        return filtered

    def _is_system_error(self, status_code: int) -> bool:
        """Проверяет, является ли ошибка системной (требует восстановления)"""
        # Только временные системные ошибки
        return status_code in [500, 502, 503, 504]

    def search(self, query: str, num_results: int = None) -> List[Dict]:
        # Проверка активации заглушки
        if self.circuit_breaker_tripped:
            self.logger.warning("Circuit breaker tripped: Skipping search for '%s'", query)
            return []
        
        # Предобработка запроса
        query = self._preprocess_query(query)
        
        num = num_results if num_results is not None else self.results_limit
        
        # Проверка кэша
        cache_key = self._get_cache_key(query, num)
        if cache_key in self.cache:
            cached_result, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                self.logger.info(f"Cache hit for query: {query}")
                return cached_result
        
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": num}
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "endpoint": self.endpoint,
            "results_limit": num
        }
        last_exception = None
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                response = requests.post(self.endpoint, headers=headers, json=payload, timeout=15)
                log_entry["http_status"] = response.status_code
                log_entry["response_text"] = response.text[:1000]  # Сокращено до 1000 символов

                # Handle 403 Forbidden (non-retriable) - отключаем заглушку навсегда
                if response.status_code == 403:
                    self.logger.error("403 Forbidden: Invalid or expired Serper API key")
                    self.logger.error(f"Response: {response.text}")
                    self._log_request_and_response(log_entry)
                    self._save_full_response(response.text, query, "403_error")
                    self.circuit_breaker_tripped = True
                    return []
                
                # Handle 429 Rate Limit (retriable)
                if response.status_code == 429:
                    last_exception = f"429 Too Many Requests (attempt {attempt+1}/{max_attempts})"
                    self.logger.warning(f"{last_exception} - Retrying...")
                    time.sleep(2 ** attempt + 1)
                    continue
                
                # Handle системные ошибки (retriable)
                if self._is_system_error(response.status_code):
                    last_exception = f"System Error {response.status_code} (attempt {attempt+1}/{max_attempts})"
                    self.logger.warning(f"{last_exception} - Retrying...")
                    time.sleep(2 ** attempt + 1)
                    continue
                
                # Handle other HTTP errors - НЕ активируем заглушку
                if not response.ok:
                    error_msg = f"HTTP Error {response.status_code}: {response.text}"
                    self.logger.error(error_msg)
                    log_entry["exception"] = error_msg
                    self._log_request_and_response(log_entry)
                    self._save_full_response(response.text, query, f"http_{response.status_code}")
                    return []
                
                # Process successful response
                data = response.json()
                if "error" in data:
                    error_msg = f"API error: {data.get('error')}"
                    self.logger.error(error_msg)
                    log_entry["error"] = error_msg
                    self._log_request_and_response(log_entry)
                    self._save_full_response(str(data), query, "api_error")
                    self.circuit_breaker_tripped = True
                    return []
                
                results = data.get("organic", [])
                
                # Сохраняем полный ответ
                self._save_full_response(response.text, query, "success")
                
                # Фильтруем результаты по качеству
                filtered_results = self._filter_quality_results(results)
                
                self.logger.info(f"Found {len(results)} search results, {len(filtered_results)} after filtering for: {query}")
                log_entry["results_count"] = len(results)
                log_entry["filtered_count"] = len(filtered_results)
                self._log_request_and_response(log_entry)
                
                # Кэшируем результат
                if filtered_results:
                    self.cache[cache_key] = (filtered_results, datetime.now())
                
                return filtered_results

            except requests.exceptions.Timeout as e:
                last_exception = f"Request Timeout: {str(e)} (attempt {attempt+1}/{max_attempts})"
                self.logger.warning(last_exception)
                time.sleep(2 ** attempt + 1)
                
            except requests.exceptions.ConnectionError as e:
                last_exception = f"Connection Error: {str(e)} (attempt {attempt+1}/{max_attempts})"
                self.logger.warning(last_exception)
                time.sleep(2 ** attempt + 1)
                
            except requests.exceptions.HTTPError as e:
                # Только для системных ошибок продолжаем попытки
                if hasattr(e, 'response') and self._is_system_error(e.response.status_code):
                    last_exception = f"System HTTP Error: {str(e)} (attempt {attempt+1}/{max_attempts})"
                    self.logger.warning(last_exception)
                    time.sleep(2 ** attempt + 1)
                    continue
                else:
                    # Для остальных HTTP ошибок - не активируем заглушку
                    error_msg = f"HTTP Error: {str(e)}"
                    self.logger.error(error_msg)
                    log_entry["exception"] = error_msg
                    self._log_request_and_response(log_entry)
                    return []
                
            except Exception as e:
                error_msg = f"Unexpected Error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                log_entry["exception"] = error_msg
                self._log_request_and_response(log_entry)
                self.circuit_breaker_tripped = True
                return []

        # All retries failed
        if last_exception:
            final_error = f"Search failed after {max_attempts} attempts: {last_exception}"
            self.logger.error(final_error)
            log_entry["exception"] = final_error
            self._log_request_and_response(log_entry)
            
            # Активируем заглушку только для системных ошибок
            if "System" in last_exception:
                self.circuit_breaker_tripped = True
                
        return []

    def _save_full_response(self, response_text: str, query: str, response_type: str) -> None:
        """Сохраняет полный ответ в txt файл"""
        folder = Path("logs/responses")
        folder.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join([c if c.isalnum() or c in " _-" else "_" for c in query])[:50]
        file_path = folder / f"{timestamp}_{safe_query}_{response_type}.txt"
        
        try:
            with file_path.open("w", encoding="utf-8") as f:
                f.write(f"Query: {query}\n")
                f.write(f"Type: {response_type}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"{'='*50}\n")
                f.write(response_text)
            self.logger.info(f"Full response saved: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save full response: {e}")

    # Все вспомогательные методы сохранены без изменений
    def build_search_query(self, topic: str) -> str:
        return topic

    def extract_content(self, search_results: List[Dict]) -> str:
        contents = [res.get("snippet", "") for res in search_results if "snippet" in res]
        return "\n\n".join(contents)

    def filter_relevant_results(self, results: List[Dict], topic: str) -> List[Dict]:
        filtered = [r for r in results if topic.lower() in r.get("title", "").lower()]
        return filtered or results

    def save_to_inform(self, content: str, topic: str, source: str = "web") -> None:
        folder = Path("inform/web")
        folder.mkdir(parents=True, exist_ok=True)
        safe_topic = "".join([c if c.isalnum() or c in " _-" else "_" for c in topic])
        file_path = folder / f"{safe_topic}_{source}.txt"
        try:
            # Append mode: create if not exists, append otherwise
            with file_path.open("a", encoding="utf-8") as f:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n\n---\nAppended at: {now}\n{content}\n")
            self.logger.info(f"Web content appended: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to append web content for {topic}", exc_info=True)

    def format_search_context(self, results: List[Dict]) -> str:
        formatted = []
        for r in results:
            url = r.get("link", "#")
            snippet = r.get("snippet", "")
            formatted.append(f"{snippet}\nИсточник: {url}")
        return "\n\n".join(formatted)

    def get_search_stats(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "results_limit": self.results_limit,
            "cache_size": len(self.cache),
            "circuit_breaker_tripped": self.circuit_breaker_tripped
        }

    def validate_search_results(self, results: List[Dict]) -> bool:
        return len(results) > 0

    def handle_rate_limits(self, response: Dict) -> bool:
        if "error" in response and "rate" in response["error"].lower():
            self.logger.warning("API rate limit reached.")
            return True
        return False

    def clean_search_content(self, content: str) -> str:
        return content.replace("\u200b", "").strip()

    def deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for r in results:
            url = r.get("link")
            if url and url not in seen:
                seen.add(url)
                unique.append(r)
        return unique

    def _log_request_and_response(self, log_entry: dict):
        """Log the search query and response to logs/web_search.log for auditing."""
        logs_folder = Path("logs")
        logs_folder.mkdir(exist_ok=True)
        log_file = logs_folder / "web_search.log"
        try:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} | QUERY: {log_entry.get('query')}\n")
                for k, v in log_entry.items():
                    if k != "query":
                        # Pretty-print long values
                        v_str = str(v)
                        if len(v_str) > 1000:  # Сокращено до 1000 символов
                            v_str = v_str[:1000] + "...[truncated]"
                        f.write(f"  {k}: {v_str}\n")
                f.write("\n")
        except Exception as e:
            self.logger.error(f"Failed to write web search log: {e}")
