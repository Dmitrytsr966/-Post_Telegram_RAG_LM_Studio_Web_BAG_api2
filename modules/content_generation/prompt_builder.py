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
        for folder in self.prompt_folders:
            if not folder.exists():
                self.logger.warning(f"Prompt folder does not exist: {folder}")
                self.templates[str(folder)] = []
                continue
            self.templates[str(folder)] = self._scan_prompt_folder(folder)
            self.logger.info(f"Loaded {len(self.templates[str(folder)])} templates from {folder}")

    def _scan_prompt_folder(self, folder_path: Path) -> List[str]:
        return [str(p) for p in folder_path.glob("*.txt")]

    def _select_random_templates(self) -> List[str]:
        selected = []
        for folder in self.prompt_folders:
            templates = self.templates.get(str(folder), [])
            selected.append(random.choice(templates) if templates else None)
        return selected

    def _read_template_file(self, file_path: Optional[str]) -> str:
        if not file_path:
            return ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            msg = f"Failed to read prompt file: {file_path}"
            self.logger.error(msg, exc_info=True)
            return ""

    def _validate_prompt_structure(self, template: str) -> None:
        missing = [ph for ph in self.REQUIRED_PLACEHOLDERS if ph not in template]
        if missing:
            raise ValueError(f"Prompt missing required placeholders: {missing}")
        for ph in self.REQUIRED_PLACEHOLDERS:
            if ph * 2 in template:
                self.logger.warning(f"Prompt contains duplicated placeholder: {ph}{ph}")
        all_placeholders = set(self.PLACEHOLDER_PATTERN.findall(template))
        supported = set(self.REQUIRED_PLACEHOLDERS)
        unsupported = [ph for ph in all_placeholders if ph not in supported]
        if unsupported:
            self.logger.warning(f"Prompt contains unsupported placeholders: {unsupported}")

    def _find_unresolved_placeholders(self, text: str) -> List[str]:
        return list(set(self.PLACEHOLDER_PATTERN.findall(text)))

    def _replace_placeholders(self, template: str, replacements: Dict[str, str]) -> str:
        for key, value in replacements.items():
            template = template.replace(key, value)
        return template

    def build_prompt(self, topic: str, context: str) -> Tuple[str, str]:
        if not topic or not isinstance(topic, str):
            raise ValueError("Topic for prompt_builder is empty or not a string.")
        if not context or not isinstance(context, str):
            raise ValueError("Context for prompt_builder is empty or not a string.")

        template_paths = self._select_random_templates()
        template_texts = [self._read_template_file(path) for path in template_paths if path]
        if not any(template_texts):
            prompt_template = self._default_template()
            self.logger.warning("No prompt templates found, using default.")
        else:
            prompt_template = "\n\n".join(filter(None, template_texts)).strip()

        self._last_prompt_template = prompt_template
        self._validate_prompt_structure(prompt_template)

        replacements = {
            "{TOPIC}": topic.strip(),
            "{CONTEXT}": context.strip(),
        }
        prompt = self._replace_placeholders(prompt_template, replacements)

        unresolved = self._find_unresolved_placeholders(prompt)
        critical_unresolved = [ph for ph in unresolved if ph in self.REQUIRED_PLACEHOLDERS]
        if critical_unresolved:
            raise ValueError(f"Prompt contains unresolved placeholders after replacement: {critical_unresolved}")

        return prompt, prompt_template

    def _default_template(self) -> str:
        return (
            "Ты опытный механик грузовой техники с 15-летним стажем работы.\n"
            "Работал в крупных автопарках, ремонтировал краны, фургоны, бортовые машины. "
            "Знаешь все подводные камни эксплуатации. Говоришь простым языком, приводишь примеры из практики.\n\n"
            "Тема: {TOPIC}\n\n"
            "Контекст для анализа: {CONTEXT}"
        )
