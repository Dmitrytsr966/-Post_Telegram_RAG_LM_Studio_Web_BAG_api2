import os
import random
import logging
import json
import time
from typing import Optional, List, Tuple
from pathlib import Path
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import textwrap

class ContentManager:
    """
    Упрощённый менеджер контента для автоматического управления медиафайлами (фото/видео)
    и оптимизации длины сообщений для Telegram.
    """
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger("ContentManager")
        self.config = config
        
        # Настройка логирования в папку logs
        self._setup_logging()
        
        self.media_dir = Path(config["paths"].get("media_dir", "media"))
        self.data_dir = Path(config["paths"].get("data_dir", "data"))
        
        self.min_text_length = config.get("content_manager", {}).get("min_text_length", 1024)
        self.max_caption_length = config["telegram"].get("max_caption_length", 1024)
        self.min_telegram_length = 712
        
        self.recent_files_file = self.data_dir / "recent_media_files.json"
        self.max_recent_files = 15
        
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        self.video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
        self.supported_extensions = self.image_extensions + self.video_extensions
        
        self.enable_media_selection = config.get("content_manager", {}).get("enable_media_selection", True)
        self.preferred_media_type = config.get("content_manager", {}).get("preferred_media_type")
        self.avoid_recent_media = config.get("content_manager", {}).get("avoid_recent_media", True)
        self.cooldown_period_hours = config.get("content_manager", {}).get("cooldown_period_hours", 1)
        self.max_media_file_size_mb = config.get("content_manager", {}).get("max_media_file_size_mb", 50)
        self.auto_cleanup_missing_files = config.get("content_manager", {}).get("auto_cleanup_missing_files", True)
        
        # === НАСТРОЙКИ ТЕКСТА И ШРИФТОВ ===
        self.font_size_range = [32, 56]  # Диапазон размеров основного шрифта
        self.text_color = (253, 244, 227)  # Цвет текста (R, G, B)
        self.text_opacity = 0.8  # Прозрачность текста (0.0-1.0)
        self.background_opacity = 0.5  # Прозрачность фона под текстом (0.0-1.0)
        self.background_color = (255, 71, 202)  # Цвет фона под текстом (R, G, B)
        self.margin_percent = 0.05  # Отступы от краев изображения (% от размера)
        self.max_text_width_percent = 0.9  # Максимальная ширина текста (% от ширины изображения)
        self.line_spacing = 17  # Расстояние между строками в пикселях
        
        # === НАСТРОЙКИ ВОДЯНОГО ЗНАКА ===
        self.watermark_text = "Работа для девушек"
        self.watermark_font_size = 45  # Размер шрифта водяного знака (увеличен)
        self.watermark_color = (253, 244, 227)  # Цвет водяного знака (R, G, B)
        self.watermark_opacity = 0.9  # Прозрачность водяного знака (0.0-1.0)
        
        self._ensure_directories()
        self._load_recent_files()
        
        if self.auto_cleanup_missing_files:
            self._cleanup_missing_files()
        
        self.logger.info(f"ContentManager initialized. Media dir: {self.media_dir}")

    def _setup_logging(self):
        """Настраивает логирование в папку logs."""
        try:
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # Создаем файловый обработчик
            log_file = logs_dir / "content_manager.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # Формат логов
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            # Добавляем обработчик к логгеру
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.DEBUG)
            
            self.logger.info("Logging initialized for ContentManager")
            
        except Exception as e:
            print(f"Failed to setup logging: {e}")

    def _ensure_directories(self):
        """Создает необходимые директории если они не существуют."""
        try:
            self.media_dir.mkdir(parents=True, exist_ok=True)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Создаем папку fonts если её нет
            fonts_dir = Path("fonts")
            fonts_dir.mkdir(exist_ok=True)
            
            # Проверяем наличие шрифтов и выводим информацию
            self._check_fonts_availability()
                
        except Exception as e:
            self.logger.error(f"Failed to create directories: {e}")
    
    def _check_fonts_availability(self):
        """Проверяет доступность шрифтов и выводит подробную информацию."""
        fonts_dir = Path("fonts")
        
        self.logger.info("=== FONT AVAILABILITY CHECK ===")
        self.logger.info(f"Fonts directory: {fonts_dir.absolute()}")
        self.logger.info(f"Fonts directory exists: {fonts_dir.exists()}")
        
        if fonts_dir.exists():
            font_files = list(fonts_dir.glob("*.*"))
            self.logger.info(f"Files in fonts directory: {[f.name for f in font_files]}")
            
            # Ищем нужные шрифты
            patsy_files = [f for f in font_files if 'patsy' in f.name.lower()]
            nexa_files = [f for f in font_files if 'nexa' in f.name.lower()]
            
            self.logger.info(f"Patsy fonts found: {[f.name for f in patsy_files]}")
            self.logger.info(f"Nexa fonts found: {[f.name for f in nexa_files]}")
        else:
            self.logger.warning("Fonts directory does not exist!")
        
        self.logger.info("=== END FONT CHECK ===")

    def _load_recent_files(self):
        """Загружает список недавно использованных файлов."""
        try:
            if self.recent_files_file.exists():
                with open(self.recent_files_file, 'r', encoding='utf-8') as f:
                    recent_list = json.load(f)
                    self.recent_files = deque(recent_list, maxlen=self.max_recent_files)
            else:
                self.recent_files = deque(maxlen=self.max_recent_files)
        except Exception as e:
            self.logger.warning(f"Failed to load recent files list: {e}")
            self.recent_files = deque(maxlen=self.max_recent_files)

    def _save_recent_files(self):
        """Сохраняет список недавно использованных файлов."""
        try:
            with open(self.recent_files_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.recent_files), f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save recent files list: {e}")

    def get_media_files(self, media_type: Optional[str] = None) -> List[Path]:
        """
        Получает список всех доступных медиафайлов.
        
        Args:
            media_type: 'image', 'video' или None для всех типов
            
        Returns:
            List[Path]: Список путей к медиафайлам
        """
        media_files = []
        
        try:
            if self.media_dir.exists():
                for file_path in self.media_dir.rglob('*'):
                    if (file_path.is_file() and 
                        file_path.suffix.lower() in self.supported_extensions and
                        self._is_valid_media_file(file_path)):
                        
                        if media_type == 'image' and file_path.suffix.lower() in self.image_extensions:
                            media_files.append(file_path)
                        elif media_type == 'video' and file_path.suffix.lower() in self.video_extensions:
                            media_files.append(file_path)
                        elif media_type is None:
                            media_files.append(file_path)
                            
        except Exception as e:
            self.logger.error(f"Failed to get media files: {e}")
            
        return media_files

    def _is_valid_media_file(self, file_path: Path) -> bool:
        """Проверяет валидность медиафайла."""
        try:
            max_size = self.max_media_file_size_mb * 1024 * 1024
            if file_path.stat().st_size > max_size:
                return False
            
            min_size = 1024
            if file_path.stat().st_size < min_size:
                return False
                
            return True
            
        except Exception:
            return False

    def select_random_media(self, media_type: Optional[str] = None) -> Optional[Path]:
        """
        Выбирает случайный медиафайл, избегая недавно использованные.
        
        Args:
            media_type: 'image', 'video' или None для любого типа
            
        Returns:
            Optional[Path]: Путь к выбранному файлу или None
        """
        if not self.enable_media_selection:
            return None
            
        try:
            filter_type = self.preferred_media_type or media_type
            all_files = self.get_media_files(filter_type)
            
            if not all_files:
                self.logger.warning(f"No media files found for type: {filter_type}")
                return None
            
            if not self.avoid_recent_media or len(all_files) <= self.max_recent_files:
                selected_file = random.choice(all_files)
                if self.avoid_recent_media:
                    self.recent_files.append(str(selected_file))
                    self._save_recent_files()
                self.logger.info(f"Selected media file (small collection): {selected_file.name}")
                return selected_file
            
            available_files = [f for f in all_files if str(f) not in self.recent_files]
            
            if not available_files:
                self.logger.warning("No available files found, clearing recent history")
                self.recent_files.clear()
                available_files = all_files
            
            selected_file = random.choice(available_files)
            
            self.recent_files.append(str(selected_file))
            self._save_recent_files()
            
            self.logger.info(f"Selected media file: {selected_file.name}")
            return selected_file
            
        except Exception as e:
            self.logger.error(f"Failed to select random media: {e}")
            return None

    def should_add_media(self, text: str) -> bool:
        """
        Определяет, нужно ли добавить медиафайл к тексту.
        
        Args:
            text: Текст сообщения
            
        Returns:
            bool: True если нужно добавить медиафайл
        """
        text_length = len(text.strip())
        
        if text_length < self.min_telegram_length:
            self.logger.info(f"Text too short for Telegram ({text_length} < {self.min_telegram_length})")
            return False
        
        if text_length < self.min_text_length:
            self.logger.info(f"Text is short ({text_length} < {self.min_text_length}), adding media")
            return True
            
        return False

    def _adapt_text_for_caption(self, text: str) -> str:
        """Адаптирует текст для использования в качестве caption."""
        if len(text) <= self.max_caption_length:
            return text
        
        sentences = text.split('.')
        adapted_text = ""
        
        for sentence in sentences:
            test_text = adapted_text + sentence + "."
            if len(test_text) <= self.max_caption_length - 50:
                adapted_text = test_text
            else:
                break
        
        if not adapted_text:
            adapted_text = text[:self.max_caption_length - 3] + "..."
        
        return adapted_text.strip()

    def _get_font_path(self, font_type: str = "main") -> str:
        """
        Получает путь к шрифту в зависимости от типа.
        Использует более гибкий подход для поиска шрифтов.
        """
        fonts_dir = Path("fonts")
        
        # Паттерны для поиска шрифтов
        if font_type == "main":
            search_patterns = [
                "*patsy*sans*",
                "*patsy*",
                "Patsy*",
                "patsy*"
            ]
        else:  # watermark
            search_patterns = [
                "*nexa*script*",
                "*nexa*",
                "Nexa*",
                "nexa*"
            ]
        
        self.logger.debug(f"Looking for {font_type} font with patterns: {search_patterns}")
        
        # Ищем по паттернам
        if fonts_dir.exists():
            for pattern in search_patterns:
                matches = list(fonts_dir.glob(pattern))
                if matches:
                    # Предпочитаем .otf, потом .ttf
                    otf_matches = [m for m in matches if m.suffix.lower() == '.otf']
                    ttf_matches = [m for m in matches if m.suffix.lower() == '.ttf']
                    
                    font_file = None
                    if otf_matches:
                        font_file = otf_matches[0]
                    elif ttf_matches:
                        font_file = ttf_matches[0]
                    elif matches:
                        font_file = matches[0]
                    
                    if font_file and font_file.exists():
                        self.logger.info(f"Found {font_type} font: {font_file}")
                        return str(font_file)
        
        # Системные fallback шрифты
        system_fonts = [
            "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\calibri.ttf",
            "C:\\Windows\\Fonts\\tahoma.ttf"
        ]
        
        for sys_font in system_fonts:
            if os.path.exists(sys_font):
                self.logger.warning(f"Using system fallback font for {font_type}: {sys_font}")
                return sys_font
        
        self.logger.error(f"No font found for {font_type}, using PIL default")
        return "arial.ttf"  # PIL fallback

    def _get_optimal_font_size(self, text: str, image_width: int, image_height: int, font_path: str) -> int:
        """Определяет оптимальный размер шрифта для заданного текста и изображения."""
        margin = int(min(image_width, image_height) * self.margin_percent)
        max_width = int(image_width * self.max_text_width_percent) - (margin * 2)
        max_height = int(image_height * 0.4)
        
        for font_size in range(self.font_size_range[1], self.font_size_range[0] - 1, -2):
            try:
                font = ImageFont.truetype(font_path, font_size)
                
                # Разбиваем текст на слова для правильного переноса
                words = text.upper().split()
                lines = []
                current_line = ""
                
                for word in words:
                    test_line = current_line + (" " if current_line else "") + word
                    test_width = font.getbbox(test_line)[2] - font.getbbox(test_line)[0]
                    
                    if test_width <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                            current_line = word
                        else:
                            # Если одно слово не помещается, принудительно разбиваем его
                            lines.append(word)
                            current_line = ""
                
                if current_line:
                    lines.append(current_line)
                
                # Проверяем общую высоту текста
                line_height = font.getbbox('A')[3] - font.getbbox('A')[1]
                total_height = len(lines) * line_height + (len(lines) - 1) * self.line_spacing
                
                if total_height <= max_height:
                    return font_size
                    
            except Exception as e:
                self.logger.debug(f"Font size {font_size} test failed: {e}")
                continue
                
        return self.font_size_range[0]

    def _add_watermark(self, draw: ImageDraw.Draw, width: int, height: int) -> None:
        """Добавляет вертикальный водяной знак в правый верхний угол."""
        try:
            watermark_font_path = self._get_font_path("watermark")
            
            try:
                watermark_font = ImageFont.truetype(watermark_font_path, self.watermark_font_size)
            except Exception as e:
                self.logger.warning(f"Failed to load watermark font: {e}")
                watermark_font = ImageFont.load_default()
            
            # Вычисляем размеры текста
            text_bbox = watermark_font.getbbox(self.watermark_text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Создаем изображение точно под размер текста
            watermark_img = Image.new('RGBA', (text_width + 20, text_height + 10), (0, 0, 0, 0))
            watermark_draw = ImageDraw.Draw(watermark_img)
            
            watermark_color_with_alpha = self.watermark_color + (int(255 * self.watermark_opacity),)
            watermark_draw.text((10, 5), self.watermark_text, 
                              fill=watermark_color_with_alpha, font=watermark_font)
            
            # Поворачиваем на 90 градусов
            rotated_watermark = watermark_img.rotate(90, expand=True)
            
            # Размещаем точно в правом верхнем углу
            paste_x = width - rotated_watermark.width - 5  # 5px от правого края
            paste_y = 5  # 5px от верхнего края
            
            temp_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            temp_img.paste(rotated_watermark, (paste_x, paste_y), rotated_watermark)
            
            return temp_img
            
        except Exception as e:
            self.logger.warning(f"Failed to add watermark: {e}")
            return None

    def _create_text_overlay(self, image_path: Path, topic: str) -> Optional[Path]:
        """
        Создает изображение с наложенным текстом темы.
        
        Args:
            image_path: Путь к исходному изображению
            topic: Тема для наложения
            
        Returns:
            Optional[Path]: Путь к обработанному изображению или None
        """
        self.logger.info(f"Creating text overlay for image: {image_path} with topic: {topic}")
        
        try:
            # Проверяем существование исходного файла
            if not image_path.exists():
                self.logger.error(f"Source image not found: {image_path}")
                return None
            
            self.logger.debug(f"Opening image: {image_path}")
            with Image.open(image_path) as img:
                self.logger.debug(f"Image opened successfully. Size: {img.size}, Mode: {img.mode}")
                
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                    self.logger.debug("Converted image to RGBA mode")
                
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                
                width, height = img.size
                margin = int(min(width, height) * self.margin_percent)
                self.logger.debug(f"Image dimensions: {width}x{height}, margin: {margin}")
                
                # Получаем шрифт
                main_font_path = self._get_font_path("main")
                font_size = self._get_optimal_font_size(topic, width, height, main_font_path)
                self.logger.debug(f"Using font size: {font_size}")
                
                try:
                    font = ImageFont.truetype(main_font_path, font_size)
                    self.logger.debug(f"Font loaded successfully: {main_font_path}")
                except Exception as e:
                    self.logger.error(f"Failed to load font {main_font_path}: {e}")
                    font = ImageFont.load_default()
                    self.logger.warning("Using default font")
                
                # Правильный перенос текста по словам с контролем ширины
                max_width = int(width * self.max_text_width_percent) - (margin * 2)
                self.logger.debug(f"Max text width: {max_width}")
                
                words = topic.upper().split()
                lines = []
                current_line = ""
                
                for word in words:
                    test_line = current_line + (" " if current_line else "") + word
                    test_width = font.getbbox(test_line)[2] - font.getbbox(test_line)[0]
                    
                    if test_width <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                            current_line = word
                        else:
                            # Если одно слово слишком длинное, всё равно добавляем его
                            lines.append(word)
                            current_line = ""
                
                if current_line:
                    lines.append(current_line)
                
                self.logger.debug(f"Text split into {len(lines)} lines: {lines}")
                
                # Вычисляем позиции для размещения текста
                line_heights = []
                for line in lines:
                    bbox = font.getbbox(line)
                    line_heights.append(bbox[3] - bbox[1])
                
                total_text_height = sum(line_heights) + (len(lines) - 1) * self.line_spacing
                start_y = height - total_text_height - margin * 2
                self.logger.debug(f"Text placement: start_y={start_y}, total_height={total_text_height}")
                
                current_y = start_y
                for i, line in enumerate(lines):
                    line_bbox = font.getbbox(line)
                    line_width = line_bbox[2] - line_bbox[0]
                    line_height = line_heights[i]
                    
                    # Фон-полоска по всей ширине изображения
                    bg_x1 = 0
                    bg_y1 = current_y - 8
                    bg_x2 = width
                    bg_y2 = current_y + line_height + 8
                    
                    bg_color_with_alpha = self.background_color + (int(255 * self.background_opacity),)
                    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=bg_color_with_alpha)
                    
                    # Центрируем текст, но учитываем отступы
                    centered_x = (width - line_width) // 2
                    text_x = max(margin, min(centered_x, width - line_width - margin))
                    
                    text_color_with_alpha = self.text_color + (int(255 * self.text_opacity),)
                    draw.text((text_x, current_y), line, fill=text_color_with_alpha, font=font)
                    
                    self.logger.debug(f"Line {i}: '{line}' at position ({text_x}, {current_y})")
                    current_y += line_height + self.line_spacing
                
                # Добавляем водяной знак
                self.logger.debug("Adding watermark...")
                watermark_overlay = self._add_watermark(draw, width, height)
                if watermark_overlay:
                    overlay = Image.alpha_composite(overlay, watermark_overlay)
                    self.logger.debug("Watermark added successfully")
                else:
                    self.logger.warning("Failed to add watermark")
                
                # Объединяем изображения
                self.logger.debug("Combining images...")
                combined = Image.alpha_composite(img, overlay)
                if combined.mode != 'RGB':
                    combined = combined.convert('RGB')
                    self.logger.debug("Converted final image to RGB")
                
                # Сохраняем результат
                output_path = self.data_dir / f"temp_overlay_{int(time.time())}.jpg"
                self.logger.debug(f"Saving result to: {output_path}")
                
                # Убеждаемся, что папка data существует
                self.data_dir.mkdir(exist_ok=True)
                
                combined.save(output_path, 'JPEG', quality=95)
                
                # Проверяем, что файл сохранился
                if output_path.exists():
                    file_size = output_path.stat().st_size
                    self.logger.info(f"Text overlay created successfully: {output_path} (size: {file_size} bytes)")
                    return output_path
                else:
                    self.logger.error(f"Failed to save overlay image: {output_path}")
                    return None
                
        except Exception as e:
            self.logger.error(f"Failed to create text overlay: {e}", exc_info=True)
            return None

    def process_content(self, text: str, topic: str, preferred_media_type: Optional[str] = None) -> Tuple[str, Optional[str], bool]:
        """
        Основной метод для обработки контента.
        
        Args:
            text: Исходный текст
            topic: Тема для наложения на изображение
            preferred_media_type: Игнорируется (оставлен для совместимости)
            
        Returns:
            Tuple[str, Optional[str], bool]: (текст, путь_к_медиа, успешность)
        """
        self.logger.info(f"=== PROCESSING CONTENT ===")
        self.logger.info(f"Text length: {len(text.strip())}")
        self.logger.info(f"Topic: {topic}")
        
        try:
            if len(text.strip()) < self.min_telegram_length:
                self.logger.error(f"Text too short for Telegram: {len(text.strip())} < {self.min_telegram_length}")
                return text, None, False
            
            if not self.should_add_media(text):
                self.logger.info("Media not needed, returning text only")
                return text, None, True
            
            media_file = self.select_random_media()
            if not media_file:
                self.logger.warning("No suitable media file found, sending text only")
                return text, None, True
            
            processed_media_path = media_file
            
            if media_file.suffix.lower() in self.image_extensions and topic:
                self.logger.info("Creating text overlay for image...")
                overlay_path = self._create_text_overlay(media_file, topic)
                if overlay_path:
                    processed_media_path = overlay_path
                    self.logger.info(f"Using image with text overlay: {topic}")
                    
                    # Дополнительная проверка файла
                    if processed_media_path.exists():
                        file_size = processed_media_path.stat().st_size
                        self.logger.info(f"Final media file: {processed_media_path} (size: {file_size} bytes)")
                    else:
                        self.logger.error(f"Final media file not found: {processed_media_path}")
                        return text, None, False
                else:
                    self.logger.warning("Failed to create text overlay, using original image")
            
            adapted_text = self._adapt_text_for_caption(text)
            
            final_media_path = str(processed_media_path)
            self.logger.info(f"Content prepared successfully!")
            self.logger.info(f"Final text length: {len(adapted_text)}")
            self.logger.info(f"Final media path: {final_media_path}")
            self.logger.info("=== END PROCESSING ===")
            
            return adapted_text, final_media_path, True
            
        except Exception as e:
            self.logger.error(f"Failed to process content: {e}", exc_info=True)
            return text, None, False

    def get_media_stats(self) -> dict:
        """Получает статистику по медиафайлам."""
        try:
            all_files = self.get_media_files()
            image_files = self.get_media_files('image')
            video_files = self.get_media_files('video')
            
            by_extension = {}
            for file_path in all_files:
                ext = file_path.suffix.lower()
                by_extension[ext] = by_extension.get(ext, 0) + 1
            
            stats = {
                "total_files": len(all_files),
                "image_files": len(image_files),
                "video_files": len(video_files),
                "by_extension": by_extension,
                "recent_files_count": len(self.recent_files),
                "available_files_count": len(all_files) - len(self.recent_files) if len(all_files) > self.max_recent_files else len(all_files)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get media stats: {e}")
            return {}

    def reset_usage_history(self):
        """Сбрасывает историю использования файлов."""
        try:
            self.recent_files.clear()
            self._save_recent_files()
            self.logger.info("Media usage history reset")
        except Exception as e:
            self.logger.error(f"Failed to reset usage history: {e}")

    def cleanup_temp_files(self):
        """Очищает временные файлы с наложенным текстом."""
        try:
            temp_pattern = "temp_overlay_*.jpg"
            for temp_file in self.data_dir.glob(temp_pattern):
                if temp_file.is_file():
                    temp_file.unlink()
                    self.logger.debug(f"Cleaned up temp file: {temp_file}")
        except Exception as e:
            self.logger.error(f"Failed to cleanup temp files: {e}")

    def _cleanup_missing_files(self):
        """Удаляет из истории файлы, которые больше не существуют."""
        try:
            if not self.recent_files:
                return
                
            existing_files = []
            for file_path_str in list(self.recent_files):
                if Path(file_path_str).exists():
                    existing_files.append(file_path_str)
                else:
                    self.logger.debug(f"Removed missing file from history: {file_path_str}")
            
            if len(existing_files) != len(self.recent_files):
                self.recent_files.clear()
                self.recent_files.extend(existing_files)
                self._save_recent_files()
                self.logger.info(f"Cleaned up {len(self.recent_files) - len(existing_files)} missing files from history")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup missing files: {e}")

    def determine_preferred_media_type(self, topic: str, content: str) -> str:
        """Оставлен для совместимости, но не используется."""
        return 'mixed'
