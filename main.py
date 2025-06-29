import sys
import signal
import time
import os

from modules.utils.config_manager import ConfigManager
from modules.utils.logs import get_logger, log_system_info
from modules.utils.state_manager import StateManager
from modules.rag_system.rag_retriever import RAGRetriever
from modules.external_apis.telegram_client import TelegramClient
from modules.external_apis.web_search import WebSearchClient
from modules.content_generation.lm_client import LMStudioClient
from modules.content_generation.prompt_builder import PromptBuilder
from modules.content_generation.content_validator import ContentValidator

class MonitoringService:
    def __init__(self, logger):
        self.topics_processed = 0
        self.topics_failed = 0
        self.logger = logger

    def log_success(self, topic):
        self.topics_processed += 1
        self.logger.info(f"[MONITOR] Topic processed: {topic}")

    def log_failure(self, topic, error):
        self.topics_failed += 1
        self.logger.error(f"[MONITOR] Topic failed: {topic}, error: {error}")

    def report(self):
        self.logger.info(
            f"[MONITOR] Stats: Success: {self.topics_processed}, Failed: {self.topics_failed}"
        )

class TelegramRAGSystem:
    def __init__(self, config_path: str = "config/config.json"):
        self.logger = get_logger("Main")
        self.logger.info("🚀 Initializing TelegramRAGSystem...")
        self.shutdown_requested = False

        try:
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.config
        except Exception as e:
            self.logger.critical("Config initialization failed", exc_info=True)
            sys.exit(1)

        self.setup_logging()
        self.validate_configuration()
        self.initialize_services()
        self.autoload_topics()

    def setup_logging(self):
        log_system_info(self.logger)

    def validate_configuration(self):
        if not self.config_manager.validate_config():
            self.logger.critical("Configuration validation failed.")
            sys.exit(1)
        self.logger.info("Configuration validated successfully.")

    def initialize_services(self):
        try:
            self.rag_retriever = RAGRetriever(config=self.config["rag"])
            self.state_manager = StateManager(state_file="data/state.json")
            self.monitoring = MonitoringService(self.logger)

            self.lm_client = LMStudioClient(
                base_url=self.config["lm_studio"]["base_url"],
                model=self.config["lm_studio"]["model"],
                config=self.config["lm_studio"],
            )
            self.prompt_builder = PromptBuilder(
                prompt_folders=self.config["paths"].get(
                    "prompt_folders",
                    ["data/prompt_1", "data/prompt_2", "data/prompt_3"]
                )
            )
            self.content_validator = ContentValidator(config=self.config)

            serper_api_key = self.config_manager.get_serper_api_key()
            serper_endpoint = self.config_manager.get_config_value(
                "serper.endpoint", "https://google.serper.dev/search"
            )
            serper_results_limit = self.config_manager.get_config_value(
                "serper.results_limit", 10
            )
            self.web_search = WebSearchClient(
                api_key=serper_api_key,
                endpoint=serper_endpoint,
                results_limit=serper_results_limit,
            )

            token = self.config_manager.get_telegram_token()
            channel_id = self.config_manager.get_telegram_channel_id()
            self.telegram_client = TelegramClient(
                token=token, channel_id=channel_id, config=self.config["telegram"]
            )
        except Exception as e:
            self.logger.critical("Component initialization failed", exc_info=True)
            sys.exit(1)

    def autoload_topics(self):
        topics_file = "data/topics.txt"
        if not os.path.isfile(topics_file):
            self.logger.warning(f"Topics file not found: {topics_file}")
            return

        try:
            with open(topics_file, "r", encoding="utf-8") as f:
                topics = [line.strip() for line in f if line.strip()]
            existing = set(
                self.state_manager.get_unprocessed_topics()
                + self.state_manager.get_processed_topics()
                + self.state_manager.get_failed_topics()
            )
            new_topics = [t for t in topics if t not in existing]
            if new_topics:
                self.logger.info(
                    f"Autoloading {len(new_topics)} new topics into queue"
                )
                self.state_manager.add_topics(new_topics)
            else:
                self.logger.info("No new topics found to autoload")
        except Exception as e:
            self.logger.error("Failed to autoload topics", exc_info=True)

    def graceful_shutdown(self, *_):
        self.shutdown_requested = True
        self.logger.warning("Shutdown signal received. Exiting loop...")

    def get_next_topic(self) -> str:
        topic = self.state_manager.get_next_unprocessed_topic()
        if topic:
            self.logger.info(f"Next topic selected: {topic}")
        else:
            self.logger.info("No more topics to process.")
        return topic

    def truncate_rag_context(self, rag_context: str, limit: int = 10000) -> str:
        if rag_context and len(rag_context) > limit:
            self.logger.warning(
                f"RAG context too long ({len(rag_context)} > {limit}), truncating."
            )
            return rag_context[:limit] + "\n... [RAG контекст обрезан до 10 000 символов]"
        return rag_context

    def combine_contexts(self, rag_context: str, web_context: str) -> str:
        if not rag_context and not web_context:
            return ""
        elif not rag_context:
            return f"[Web context only]\n\n{web_context}"
        elif not web_context:
            return f"{rag_context}\n\n[Нет web-контекста]"
        return f"{rag_context}\n\n[Доп. контекст из поиска]\n\n{web_context}"

    def update_processing_state(self, topic: str, success: bool):
        try:
            self.state_manager.mark_topic_processed(topic, success)
            self.logger.info(
                f"Topic '{topic}' marked as {'processed' if success else 'failed'}."
            )
        except Exception as e:
            self.logger.error(
                f"Failed to update state for topic '{topic}': {str(e)}", exc_info=True
            )

    def handle_error(self, topic: str, error: Exception):
        try:
            self.logger.error(
                f"Error processing topic '{topic}': {str(error)}", exc_info=True
            )
            self.update_processing_state(topic, success=False)
            self.monitoring.log_failure(topic, error)
        except Exception as e:
            self.logger.critical("Failed during error handling!", exc_info=True)

    def main_processing_loop(self):
        self.logger.info("Entering main processing loop.")
        while not self.shutdown_requested:
            topic = self.get_next_topic()
            if not topic:
                break

            try:
                self.logger.info(f"Processing topic: {topic}")

                rag_context = self.rag_retriever.retrieve_context(topic)
                rag_context = self.truncate_rag_context(rag_context, 10000)

                if not isinstance(rag_context, str) or not rag_context.strip():
                    self.logger.error(f"RAG context is empty for topic: {topic}")
                    self.update_processing_state(topic, success=False)
                    self.monitoring.log_failure(topic, "RAG context is empty")
                    continue

                web_results = self.web_search.search(topic)
                web_context = self.web_search.extract_content(web_results) if web_results else ""
                if not isinstance(web_context, str):
                    web_context = ""

                full_context = self.combine_contexts(rag_context, web_context)
                self.logger.debug(
                    f"[{topic}] full_context length: {len(full_context)}, preview: {full_context[:300]}"
                )

                prompt, prompt_template = self.prompt_builder.build_prompt(
                    topic=topic, context=full_context
                )

                if not prompt or not prompt.strip():
                    self.logger.error(
                        f"Prompt building failed (empty) for topic '{topic}'."
                    )
                    self.update_processing_state(topic, success=False)
                    self.monitoring.log_failure(
                        topic, "Prompt building failed (empty prompt)"
                    )
                    continue

                self.logger.debug(
                    f"Prompt to LM Studio for topic '{topic}':\n{prompt[:1000]}"
                )

                max_lm_retries = self.config["lm_studio"].get("max_retries", 3)
                try:
                    content = self.lm_client.generate_with_retry(
                        prompt_template,
                        topic,
                        full_context,
                        max_retries=max_lm_retries,
                    )
                except Exception as e:
                    self.logger.error(
                        f"LM Studio generation failed after retries for topic '{topic}': {e}"
                    )
                    self.update_processing_state(topic, success=False)
                    self.monitoring.log_failure(
                        topic, f"LM Studio generation failed: {e}"
                    )
                    continue

                if not content or not content.strip():
                    self.logger.error(
                        f"Generated content is empty for topic '{topic}'."
                    )
                    self.update_processing_state(topic, success=False)
                    self.monitoring.log_failure(
                        topic, "Generated content is empty"
                    )
                    continue

                # Always validate content before sending to Telegram (extra safety)
                validated_content = self.content_validator.validate_content(content)
                if not validated_content or not validated_content.strip():
                    self.logger.error(
                        f"Validated content is empty for topic '{topic}'."
                    )
                    self.update_processing_state(topic, success=False)
                    self.monitoring.log_failure(
                        topic, "Validated content is empty"
                    )
                    continue

                success = False
                max_retries = self.config["telegram"].get("max_retries", 3)
                for attempt in range(1, max_retries + 1):
                    try:
                        self.logger.info(f"Telegram send attempt {attempt} for topic: {topic}")
                        success = self.telegram_client.send_text_message(validated_content)
                        if success:
                            self.logger.info(
                                f"Successfully sent topic '{topic}' to Telegram on attempt {attempt}."
                            )
                            break
                        else:
                            self.logger.warning(
                                f"Telegram send failed for topic '{topic}' on attempt {attempt}."
                            )
                    except Exception as te:
                        self.logger.error(
                            f"Telegram send failed (attempt {attempt}): {te}", exc_info=True
                        )
                        time.sleep(2)

                self.update_processing_state(topic, success)
                if success:
                    self.monitoring.log_success(topic)
                else:
                    self.monitoring.log_failure(topic, "Telegram send failed")

                self.monitoring.report()
                # Pause between posts according to config to avoid spam/ban
                time.sleep(self.config["telegram"].get("post_interval", 15))

            except Exception as e:
                self.handle_error(topic, e)
                continue

    def run(self):
        self.logger.info("System starting up...")
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)
        try:
            inform_folder = self.config["rag"].get("inform_folder", "inform/")
            self.rag_retriever.process_inform_folder(inform_folder)
            self.rag_retriever.build_knowledge_base()
        except Exception as e:
            self.logger.critical(
                f"Failed to build RAG knowledge base: {e}", exc_info=True
            )
            sys.exit(1)
        self.main_processing_loop()
        self.logger.info("System shut down gracefully.")

if __name__ == "__main__":
    system = TelegramRAGSystem()
    system.run()
