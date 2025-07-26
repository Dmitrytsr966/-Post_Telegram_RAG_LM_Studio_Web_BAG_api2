import os
import random
import logging
import json
import time
from typing import Optional, List, Tuple
from pathlib import Path
from collections import deque

class ContentManager:
    """
    Упрощённый менеджер контента для автоматического управления медиафайлами (фото/видео)
    и оптимизации длины сообщений для Telegram.
    """
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger("ContentManager")
        self.config = config
        
        # Пути из конфигурации
        self.media_dir = Path(config["paths"].get("media_dir", "media"))
        self.data_dir = Path(config["paths"].get("data_dir", "data"))
        
        # Настройки из конфигурации
        self.min_text_length = config.get("content_manager", {}).get("min_text_length", 1024)
        self.max_caption_length = config["telegram"].get("max_caption_length", 1024)
        self.min_telegram_length = 712  # Минимальная длина для Telegram
        
        # Файл для отслеживания недавно использованных файлов
        self.recent_files_file = self.data_dir / "recent_media_files.json"
        self.max_recent_files = 15  # Не повторять файл пока не пройдет 15 других
        
        # Поддерживаемые типы медиафайлов (все вперемешку)
        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.mp4', '.mov', '.avi', '.mkv', '.webm']
        
        # Инициализация
        self._ensure_directories()
        self._load_recent_files()
        
        self.logger.info(f"ContentManager initialized. Media dir: {self.media_dir}")

    def _ensure_directories(self):
        """Создает необходимые директории если они не существуют."""
        try:
            self.media_dir.mkdir(parents=True, exist_ok=True)
            self.data_dir.mkdir(parents=True, exist_ok=True)
                
        except Exception as e:
            self.logger.error(f"Failed to create directories: {e}")

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

    def get_media_files(self) -> List[Path]:
        """
        Получает список всех доступных медиафайлов (фото и видео вперемешку).
        
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
                        media_files.append(file_path)
                            
        except Exception as e:
            self.logger.error(f"Failed to get media files: {e}")
            
        return media_files

    def _is_valid_media_file(self, file_path: Path) -> bool:
        """Проверяет валидность медиафайла."""
        try:
            # Проверяем размер файла (Telegram лимит ~50MB)
            max_size = 50 * 1024 * 1024  # 50MB
            if file_path.stat().st_size > max_size:
                return False
            
            # Проверяем минимальный размер (исключаем слишком маленькие файлы)
            min_size = 1024  # 1KB
            if file_path.stat().st_size < min_size:
                return False
                
            return True
            
        except Exception:
            return False

    def select_random_media(self) -> Optional[Path]:
        """
        Выбирает случайный медиафайл, избегая недавно использованные (последние 15).
        
        Returns:
            Optional[Path]: Путь к выбранному файлу или None
        """
        try:
            all_files = self.get_media_files()
            
            if not all_files:
                self.logger.warning("No media files found")
                return None
            
            # Если файлов меньше чем лимит недавних, просто выбираем случайный
            if len(all_files) <= self.max_recent_files:
                selected_file = random.choice(all_files)
                self.recent_files.append(str(selected_file))
                self._save_recent_files()
                self.logger.info(f"Selected media file (small collection): {selected_file.name}")
                return selected_file
            
            # Фильтруем недавно использованные файлы
            available_files = [f for f in all_files if str(f) not in self.recent_files]
            
            if not available_files:
                # Если нет доступных файлов (что теоретически невозможно при правильной логике),
                # очищаем историю и выбираем случайный
                self.logger.warning("No available files found, clearing recent history")
                self.recent_files.clear()
                available_files = all_files
            
            # Выбираем случайный файл из доступных
            selected_file = random.choice(available_files)
            
            # Добавляем в очередь недавно использованных
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
        
        # Проверяем минимальную длину для Telegram
        if text_length < self.min_telegram_length:
            self.logger.info(f"Text too short for Telegram ({text_length} < {self.min_telegram_length})")
            return False
        
        # Добавляем медиа если текст короткий
        if text_length < self.min_text_length:
            self.logger.info(f"Text is short ({text_length} < {self.min_text_length}), adding media")
            return True
            
        return False

    def _adapt_text_for_caption(self, text: str) -> str:
        """Адаптирует текст для использования в качестве caption."""
        if len(text) <= self.max_caption_length:
            return text
        
        # Обрезаем по предложениям
        sentences = text.split('.')
        adapted_text = ""
        
        for sentence in sentences:
            test_text = adapted_text + sentence + "."
            if len(test_text) <= self.max_caption_length - 50:  # Оставляем запас
                adapted_text = test_text
            else:
                break
        
        if not adapted_text:
            # Принудительная обрезка
            adapted_text = text[:self.max_caption_length - 3] + "..."
        
        return adapted_text.strip()

    def process_content(self, text: str, preferred_media_type: Optional[str] = None) -> Tuple[str, Optional[str], bool]:
        """
        Основной метод для обработки контента.
        
        Args:
            text: Исходный текст
            preferred_media_type: Игнорируется (оставлен для совместимости)
            
        Returns:
            Tuple[str, Optional[str], bool]: (текст, путь_к_медиа, успешность)
        """
        try:
            # Проверяем минимальную длину для Telegram
            if len(text.strip()) < self.min_telegram_length:
                self.logger.error(f"Text too short for Telegram: {len(text.strip())} < {self.min_telegram_length}")
                return text, None, False
            
            # Проверяем, нужно ли добавлять медиа
            if not self.should_add_media(text):
                return text, None, True
            
            # Выбираем медиафайл (хаотично, любой тип)
            media_file = self.select_random_media()
            if not media_file:
                self.logger.warning("No suitable media file found, sending text only")
                return text, None, True
            
            # Адаптируем текст для caption
            adapted_text = self._adapt_text_for_caption(text)
            
            self.logger.info(f"Content prepared with media: {media_file.name}")
            return adapted_text, str(media_file), True
            
        except Exception as e:
            self.logger.error(f"Failed to process content: {e}")
            return text, None, False

    def get_media_stats(self) -> dict:
        """Получает статистику по медиафайлам."""
        try:
            all_files = self.get_media_files()
            
            # Подсчитываем файлы по расширениям
            by_extension = {}
            for file_path in all_files:
                ext = file_path.suffix.lower()
                by_extension[ext] = by_extension.get(ext, 0) + 1
            
            stats = {
                "total_files": len(all_files),
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

    def determine_preferred_media_type(self, topic: str, content: str) -> str:
        """Оставлен для совместимости, но не используется."""
        return 'mixed'  # Всегда возвращаем mixed, так как теперь все файлы вперемешку
