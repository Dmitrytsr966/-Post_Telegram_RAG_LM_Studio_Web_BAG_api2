import os
import random
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

class PromptBuilder:
    REQUIRED_PLACEHOLDERS = ["{TOPIC}", "{CONTEXT}"]
    PLACEHOLDER_PATTERN = re.compile(r"\{[A-Z_]+\}")

    def __init__(self, prompt_folders: List[str]):
        self.prompt_folders = [Path(folder) for folder in prompt_folders]
        self.logger = logging.getLogger("PromptBuilder")
        self.templates: Dict[str, List[str]] = {}
        self._last_prompt_template: Optional[str] = None
        self.load_prompt_templates()

    def load_prompt_templates(self) -> None:
        """
        Загружает все шаблоны из указанных папок и логирует количество найденных файлов.
        """
        for folder in self.prompt_folders:
            try:
                if not folder.exists():
                    self.logger.warning(f"Prompt folder does not exist: {folder}")
                    self.templates[str(folder)] = []
                    continue
                self.templates[str(folder)] = self._scan_prompt_folder(folder)
                self.logger.info(f"Loaded {len(self.templates[str(folder)])} templates from {folder}")
            except Exception as e:
                self.logger.error(f"Failed to scan prompt folder '{folder}': {e}", exc_info=True)
                self.templates[str(folder)] = []

    def _scan_prompt_folder(self, folder_path: Path) -> List[str]:
        """
        Возвращает список путей к .txt шаблонам в папке.
        """
        try:
            template_files = [str(p) for p in folder_path.glob("*.txt")]
            if not template_files:
                self.logger.warning(f"No prompt templates found in folder: {folder_path}")
            return template_files
        except Exception as e:
            self.logger.error(f"Error scanning folder {folder_path}: {e}", exc_info=True)
            return []

    def _select_random_templates(self) -> List[str]:
        """
        Случайно выбирает по одному шаблону из каждой папки.
        """
        selected = []
        for folder in self.prompt_folders:
            templates = self.templates.get(str(folder), [])
            if not templates:
                self.logger.warning(f"No templates to select in: {folder}")
                selected.append(None)
            else:
                path = random.choice(templates)
                self.logger.debug(f"Selected template '{path}' from folder '{folder}'")
                selected.append(path)
        return selected

    def _read_template_file(self, file_path: Optional[str]) -> str:
        """
        Читает содержимое файла шаблона, логирует ошибки при необходимости.
        """
        if not file_path:
            return ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.logger.debug(f"Read prompt template file: {file_path}")
            return content
        except Exception as e:
            msg = f"Failed to read prompt file: {file_path}"
            self.logger.error(msg, exc_info=True)
            return ""

    def _validate_prompt_structure(self, template: str) -> None:
        """
        Проверяет, что в шаблоне присутствуют все обязательные плейсхолдеры,
        логирует найденные проблемы и предупреждения.
        """
        missing = [ph for ph in self.REQUIRED_PLACEHOLDERS if ph not in template]
        if missing:
            self.logger.error(f"Prompt missing required placeholders: {missing}")
            raise ValueError(f"Prompt missing required placeholders: {missing}")
        for ph in self.REQUIRED_PLACEHOLDERS:
            if (ph * 2) in template:
                self.logger.warning(f"Prompt contains duplicated placeholder: {ph}{ph}")
        all_placeholders = set(self.PLACEHOLDER_PATTERN.findall(template))
        supported = set(self.REQUIRED_PLACEHOLDERS)
        unsupported = [ph for ph in all_placeholders if ph not in supported]
        if unsupported:
            self.logger.warning(f"Prompt contains unsupported placeholders: {unsupported}")

    def _find_unresolved_placeholders(self, text: str) -> List[str]:
        """
        Находит все незамещённые плейсхолдеры в тексте.
        """
        unresolved = list(set(self.PLACEHOLDER_PATTERN.findall(text)))
        if unresolved:
            self.logger.debug(f"Unresolved placeholders found: {unresolved}")
        return unresolved

    def _replace_placeholders(self, template: str, replacements: Dict[str, str]) -> str:
        """
        Заменяет плейсхолдеры на реальные значения.
        """
        for key, value in replacements.items():
            template = template.replace(key, value)
        return template

    def build_prompt(self, topic: str, context: str) -> Tuple[str, str]:
        """
        Основной метод для сборки и валидации промпта.
        Возвращает tuple (готовый промпт, исходный шаблон).
        """
        if not topic or not isinstance(topic, str):
            self.logger.error("Topic for prompt_builder is empty or not a string.")
            raise ValueError("Topic for prompt_builder is empty or not a string.")
        if not context or not isinstance(context, str):
            self.logger.error("Context for prompt_builder is empty or not a string.")
            raise ValueError("Context for prompt_builder is empty or not a string.")

        template_paths = self._select_random_templates()
        template_texts = [self._read_template_file(path) for path in template_paths if path]
        if not any(template_texts):
            prompt_template = self._default_template()
            self.logger.warning("No prompt templates found, using default.")
        else:
            prompt_template = "\n\n".join(filter(None, template_texts)).strip()

        self._last_prompt_template = prompt_template
        try:
            self._validate_prompt_structure(prompt_template)
        except ValueError as e:
            self.logger.error(f"Prompt structure validation failed: {e}")
            raise

        replacements = {
            "{TOPIC}": topic.strip(),
            "{CONTEXT}": context.strip(),
        }
        prompt = self._replace_placeholders(prompt_template, replacements)

        unresolved = self._find_unresolved_placeholders(prompt)
        critical_unresolved = [ph for ph in unresolved if ph in self.REQUIRED_PLACEHOLDERS]
        if critical_unresolved:
            self.logger.error(f"Prompt contains unresolved placeholders after replacement: {critical_unresolved}")
            raise ValueError(f"Prompt contains unresolved placeholders after replacement: {critical_unresolved}")

        self.logger.info("Prompt built successfully.")
        self.logger.debug(f"Prompt preview: {prompt[:300]}")
        return prompt, prompt_template

    def _default_template(self) -> str:
        """
        Резервный шаблон по умолчанию.
        """
        return (
            "Ты опытный механик грузовой техники с 15-летним стажем работы.\n"
            "Работал в крупных автопарках, ремонтировал краны, фургоны, бортовые машины. "
            "Знаешь все подводные камни эксплуатации. Говоришь простым языком, приводишь примеры из практики.\n\n"
            "Тема: {TOPIC}\n\n"
            "Контекст для анализа: {CONTEXT}"
        )
