import logging
import re
from typing import Dict, Optional
import emoji

class ContentValidator:
    TELEGRAM_TEXT_LIMIT = 4096
    TELEGRAM_SAFE_LIMIT = 4000
    MIN_CONTENT_LENGTH = 15

    # Только теги, поддерживаемые Telegram: https://core.telegram.org/bots/api#formatting-options
    ALLOWED_TAGS = {
        "b", "strong", "i", "em", "u", "ins", "s", "strike", "del", "code", "pre", "a"
    }

    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        self._init_patterns()
        self._load_config_overrides()

    def _load_config_overrides(self):
        self.TELEGRAM_TEXT_LIMIT = int(self.config.get("content_validator", {}).get("max_length_no_media", self.TELEGRAM_TEXT_LIMIT))
        self.TELEGRAM_SAFE_LIMIT = int(self.config.get("content_validator", {}).get("max_length_with_media", self.TELEGRAM_SAFE_LIMIT))

    def _init_patterns(self):
        self.re_tag = re.compile(r'</?([a-zA-Z0-9]+)[^>]*>')
        self.re_table_md = re.compile(
            r'(?:\|[^\n|]+\|[^\n]*\n)+(?:\|[-:| ]+\|[^\n]*\n)+(?:\|[^\n|]+\|[^\n]*\n?)+', re.MULTILINE
        )
        self.re_table_html = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE)
        self.re_think = re.compile(r'<\s*think[^>]*>.*?<\s*/\s*think\s*>', re.IGNORECASE | re.DOTALL)
        self.re_think_frag = re.compile(r'(?i)(размышления:|---|думаю:|#\s*think\s*|^\s*think\s*:)', re.MULTILINE)
        self.re_null = re.compile(r'\b(nan|None|null|NULL)\b', re.I)
        self.re_unicode = re.compile(r'[\u200b-\u200f\u202a-\u202e]+')
        self.re_hex = re.compile(r'\\x[0-9a-fA-F]{2}')
        self.re_unicode_hex = re.compile(r'_x[0-9A-Fa-f]{4}_')
        self.re_html_entity = re.compile(r'&[a-zA-Z0-9#]+;')
        self.re_spaces = re.compile(r' {3,}')
        # Обновленный паттерн для инвалидных символов - исключаем эмодзи
        self.re_invalid = re.compile(r'[^\x09\x0A\x0D\x20-\x7Eа-яА-ЯёЁa-zA-Z0-9.,:;!?()\[\]{}<>@#%^&*_+=/\\|\'\"`~$№\-\u2600-\u26FF\u2700-\u27BF\u1F600-\u1F64F\u1F300-\u1F5FF\u1F680-\u1F6FF\u1F1E0-\u1F1FF\u2000-\u206F\u2070-\u209F\u20A0-\u20CF\u2100-\u214F\u2150-\u218F\u2190-\u21FF\u2200-\u22FF\u2300-\u23FF\u2400-\u243F\u2440-\u245F\u2460-\u24FF\u2500-\u257F\u2580-\u259F\u25A0-\u25FF\u2600-\u26FF\u2700-\u27BF]')
        self.re_dots = re.compile(r'\.{3,}')
        self.re_commas = re.compile(r',,+')
        self.re_js_links = re.compile(r'\[([^\]]+)\]\((javascript|data):[^\)]+\)', re.I)
        self.re_multi_spaces = re.compile(r' {2,}')
        self.re_multi_newline = re.compile(r'\n{3,}', re.MULTILINE)
        self.re_repeated_chars = re.compile(r'(.)\1{10,}')
        # Меняем: только удаляем ## и ### (и ####) в начале строки, не всю строку
        self.re_md_heading = re.compile(r'(^[ \t]*#{2,4}[ \t]*)', re.MULTILINE)
        self.re_latex_block = re.compile(r"\$\$([\s\S]*?)\$\$", re.MULTILINE)
        self.re_latex_inline = re.compile(r"\$([^\$]+?)\$", re.DOTALL)
        
        # Markdown to Telegram HTML patterns - улучшенные для корректной обработки
        self.re_md_code_block = re.compile(r'```(.*?)```', re.DOTALL)
        self.re_md_inline_code = re.compile(r'`([^`\n]+)`')
        
        # ИСПРАВЛЕНИЕ: Простое решение - расширяем паттерн для поддержки любых символов кроме * и переносов
        # Обработка *** (жирный курсив) - должна быть первой
        self.re_md_bold_italic = re.compile(r'(?<!\*)\*\*\*([^\*\n\r]+?)\*\*\*(?!\*)')
        
        # Обработка ** (жирный) - с более точными границами  
        self.re_md_bold1 = re.compile(r'(?<!\*)\*\*([^\*\n\r]+?)\*\*(?!\*)')
        self.re_md_bold2 = re.compile(r'__([^_\n]+?)__')
        
        # Обработка * (курсив) - с более точными границами
        self.re_md_italic1 = re.compile(r'(?<!\*)\*([^\*\n\r]+?)\*(?!\*)')
        self.re_md_italic2 = re.compile(r'(?<!_)_([^_\n]+?)_(?!_)')
        
        # Остальные паттерны
        self.re_md_strike = re.compile(r'~~([^~\n]+?)~~')
        self.re_md_url = re.compile(r'\[([^\]]+)\]\((https?://[^\)]+)\)')
        
        # Паттерн для удаления пробелов в начале и конце строк
        self.re_line_spaces = re.compile(r'^[ \t]+|[ \t]+$', re.MULTILINE)
        
        # НОВОЕ: Паттерн для поиска сломанных markdown паттернов с эмодзи
        self.re_broken_md = re.compile(r'\*\*([^*\n]*?[\u2600-\u26FF\u2700-\u27BF\u1F300-\u1F5FF\u1F600-\u1F64F\u1F680-\u1F6FF\u1F1E0-\u1F1FF][^*\n]*?)\*([^*\n]*?)\*\*\*')

    def validate_content(self, text: str) -> str:
        if not isinstance(text, str):
            self.logger.error("Content validation input is not a string")
            return ""
        text = text.strip()
        if not text:
            self.logger.warning("Empty content provided for validation")
            return ""

        # 1. Удалить размышления (think) и вариации
        text = self.remove_thinking_blocks(text)
        text = self._remove_think_variations(text)
        # 2. Преобразовать markdown-разметку в Telegram HTML
        text = self.convert_markdown_to_telegram_html(text)
        # 3. Оставить только разрешённые html-теги Telegram
        text = self._remove_forbidden_html_tags(text)
        # 4. Удалить таблицы, если вдруг остались
        text = self._remove_tables_and_thinking(text)
        # 5. Прочая чистка (включая latex и markdown-заголовки)
        text = self._clean_junk(text)
        # 6. Удалить пробелы в начале и конце строк
        text = self._clean_line_spaces(text)
        # 7. Ограничить длину для Telegram
        text = self._ensure_telegram_limits(text)

        if not self._content_quality_check(text):
            self.logger.warning("Content failed quality validation")
            return ""
        return text.strip()

    def remove_thinking_blocks(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = self.re_think.sub('', text)
        return text

    def _remove_think_variations(self, text: str) -> str:
        text = self.re_think_frag.sub('', text)
        return text

    def convert_markdown_to_telegram_html(self, text: str) -> str:
        """
        Преобразует markdown в HTML, поддерживаемый Telegram.
        ВАЖНО: порядок обработки имеет значение!
        ИСПРАВЛЕНИЕ: добавлена предобработка для сломанных паттернов с эмодзи
        """
        # НОВОЕ: Исправляем сломанные паттерны перед основной обработкой
        def fix_broken_pattern(match):
            # Объединяем все части в одну для правильной обработки
            full_content = match.group(1) + match.group(2)
            self.logger.info(f"Fixing broken markdown with emoji: {match.group(0)[:50]}...")
            return f"***{full_content}***"
        
        text = self.re_broken_md.sub(fix_broken_pattern, text)
        
        # 1. Сначала обрабатываем ссылки
        text = self.re_md_url.sub(r'<a href="\2">\1</a>', text)
        
        # 2. Затем блоки кода (чтобы не затронуть markdown внутри)
        text = self.re_md_code_block.sub(lambda m: f"<pre>{self.escape_html(m.group(1).strip())}</pre>", text)
        text = self.re_md_inline_code.sub(lambda m: f"<code>{self.escape_html(m.group(1).strip())}</code>", text)
        
        # 3. Обрабатываем *** (жирный курсив) ПЕРВЫМ - самый специфичный паттерн
        text = self.re_md_bold_italic.sub(r'<b><i>\1</i></b>', text)
        
        # 4. Затем ** (жирный)
        text = self.re_md_bold1.sub(r'<b>\1</b>', text)
        text = self.re_md_bold2.sub(r'<b>\1</b>', text)
        
        # 5. Потом * (курсив)
        text = self.re_md_italic1.sub(r'<i>\1</i>', text)
        text = self.re_md_italic2.sub(r'<i>\1</i>', text)
        
        # 6. Зачеркивание
        text = self.re_md_strike.sub(r'<s>\1</s>', text)
        
        return text

    def escape_html(self, text: str) -> str:
        return (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
        )

    def _remove_forbidden_html_tags(self, text: str) -> str:
        def _strip_tag(m):
            tag = m.group(1).lower()
            if tag in self.ALLOWED_TAGS:
                return m.group(0)
            return ''
        return self.re_tag.sub(_strip_tag, text)

    def _remove_tables_and_thinking(self, text: str) -> str:
        text = self.re_table_md.sub('', text)
        text = self.re_table_html.sub('', text)
        text = self.re_think.sub('', text)
        return text

    def _clean_junk(self, text: str) -> str:
        # 1. Удаляем только ##, ###, #### (и пробелы/табы перед ними) в начале строки, не всю строку
        def log_and_sub_heading(m):
            if m.group(0):
                self.logger.info(f"Удалена решетка: '{m.group(0)}'")
            return ''
        text = self.re_md_heading.sub(log_and_sub_heading, text)
        # 2. Удаляем LaTeX-блоки и inline-$, оставляем формулу без маркеров
        text = self.re_latex_block.sub(lambda m: m.group(1).strip(), text)
        text = self.re_latex_inline.sub(lambda m: m.group(1).strip(), text)
        # 3. Прочая очистка
        text = self.re_null.sub('', text)
        text = self.re_unicode.sub('', text)
        text = self.re_hex.sub('', text)
        text = self.re_unicode_hex.sub('', text)
        text = self.re_html_entity.sub('', text)
        text = self.re_spaces.sub('  ', text)
        
        # 4. Очистка инвалидных символов с сохранением эмодзи
        text = self._clean_invalid_chars_preserve_emoji(text)
        
        text = self.re_dots.sub('…', text)
        text = self.re_commas.sub(',', text)
        text = self.re_js_links.sub(r'\1', text)
        text = self.re_multi_spaces.sub(' ', text)
        text = self.re_multi_newline.sub('\n\n', text)
        return text.strip()

    def _clean_invalid_chars_preserve_emoji(self, text: str) -> str:
        """
        Очищает инвалидные символы, но сохраняет эмодзи.
        ИСПРАВЛЕНО: правильное использование emoji.replace_emoji()
        """
        try:
            # Сначала извлекаем все эмодзи из текста
            emojis = []
            emoji_placeholder = "§EMOJI§"
        
            def extract_emoji(emoji_char, emoji_data):
                """Функция для извлечения эмодзи - принимает два аргумента"""
                emojis.append(emoji_char)
                return f"{emoji_placeholder}{len(emojis)-1}{emoji_placeholder}"
        
            # Извлекаем эмодзи - исправленный синтаксис
            text = emoji.replace_emoji(text, replace=extract_emoji)
        
            # Очищаем инвалидные символы
            text = self.re_invalid.sub('', text)
        
            # Возвращаем эмодзи обратно
            for i, emoji_char in enumerate(emojis):
                text = text.replace(f"{emoji_placeholder}{i}{emoji_placeholder}", emoji_char)
        
            return text
        
        except Exception as e:
            self.logger.error(f"Error in emoji processing: {e}")
            # Fallback: просто удаляем инвалидные символы без обработки эмодзи
            return self.re_invalid.sub('', text)

    def _clean_line_spaces(self, text: str) -> str:
        """Удаляет пробелы и табы в начале и конце строк, сохраняя переносы"""
        return self.re_line_spaces.sub('', text)

    def _ensure_telegram_limits(self, text: str) -> str:
        if len(text) <= self.TELEGRAM_TEXT_LIMIT:
            return text
        cut = self.TELEGRAM_SAFE_LIMIT
        for i in range(cut - 100, cut):
            if i < len(text) and text[i] in [".", "!", "?", "\n\n"]:
                cut = i + 1
                break
        truncated = text[:cut].rstrip()
        if not truncated.endswith(('...', '…')):
            truncated += '…'
        return truncated

    def _content_quality_check(self, text: str) -> bool:
        if not text or len(text) < self.MIN_CONTENT_LENGTH:
            return False
        word_count = len(re.findall(r'\w+', text))
        if word_count < 3:
            return False
        if self.re_repeated_chars.search(text):
            return False
        return True
