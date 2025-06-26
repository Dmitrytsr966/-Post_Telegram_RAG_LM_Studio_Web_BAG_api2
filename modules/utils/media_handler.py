import logging
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image, UnidentifiedImageError
import mimetypes

class MediaHandler:
    """
    Класс для работы с медиафайлами (изображения, видео, документы) с поддержкой валидации,
    анализа, ресайза, сбора статистики и очистки временных файлов.
    """

    def __init__(self, media_folder: str, config: dict, logger: Optional[logging.Logger] = None):
        """
        :param media_folder: Путь к папке с медиафайлами
        :param config: Конфиг с лимитами и поддерживаемыми форматами
        :param logger: Логгер, если не задан — создаётся локальный
        """
        self.media_folder = Path(media_folder)
        self.config = config
        self.logger = logger or logging.getLogger("MediaHandler")

        self.max_file_size_mb = (
            config.get("processing", {}).get("max_file_size_mb", 50)
            or config.get("max_file_size_mb", 50)
        )

        self.supported_formats = config.get("supported_formats", {
            "images": [".jpg", ".jpeg", ".png"],
            "videos": [".mp4", ".mov"],
            "documents": [".pdf", ".docx", ".txt"]
        })

        self.max_image_size = config.get("max_image_size", (1280, 1280))

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Вернуть поддерживаемые форматы медиафайлов."""
        return self.supported_formats

    def get_media_files_list(self) -> List[Path]:
        """Получить список всех поддерживаемых медиафайлов."""
        files = []
        for group in self.supported_formats.values():
            for ext in group:
                files.extend(self.media_folder.glob(f"*{ext}"))
        valid_files = [f for f in files if self.validate_media_file(f)]
        if not valid_files:
            self.logger.warning(f"No valid media files found in {self.media_folder}.")
        return valid_files

    def get_random_media_file(self) -> Optional[str]:
        """Получить случайный валидный медиафайл."""
        media_files = self.get_media_files_list()
        if not media_files:
            self.logger.warning("No media files available for selection.")
            return None
        selected = random.choice(media_files)
        self.logger.info(f"Selected media file: {selected}")
        return str(selected)

    def validate_media_file(self, file_path: Any) -> bool:
        """
        Проверить, что файл существует, не превышает max_file_size_mb, имеет поддерживаемый формат, не битый (для изображений).
        """
        try:
            p = Path(file_path)
            if not p.is_file():
                self.logger.warning(f"Media file not found: {file_path}")
                return False
            if p.stat().st_size > self.max_file_size_mb * 1024 * 1024:
                self.logger.warning(f"Media file too large: {file_path}")
                return False
            ext = p.suffix.lower()
            if not any(ext in group for group in self.supported_formats.values()):
                self.logger.warning(f"Unsupported media format: {file_path}")
                return False
            if ext in self.supported_formats["images"]:
                try:
                    with Image.open(p) as img:
                        img.verify()
                except (UnidentifiedImageError, Exception) as e:
                    self.logger.warning(f"Invalid image file: {file_path}, error: {e}")
                    return False
            # TODO: добавить проверки для других типов файлов при необходимости
            return True
        except Exception as e:
            self.logger.error(f"Failed to validate media file: {file_path}, error: {e}")
            return False

    def get_media_type(self, file_path: str) -> str:
        """
        Определить тип медиа (image, video, document) по расширению.
        """
        ext = Path(file_path).suffix.lower()
        for typ, group in self.supported_formats.items():
            if ext in group:
                return typ
        self.logger.warning(f"Unknown media extension: {file_path}")
        return "documents"

    def get_file_size(self, file_path: str) -> int:
        """Получить размер файла в байтах."""
        try:
            return Path(file_path).stat().st_size
        except Exception as e:
            self.logger.error(f"Failed to get file size: {file_path}, error: {e}")
            return -1

    def resize_image_if_needed(self, file_path: str, max_size: Tuple[int, int] = None) -> Optional[str]:
        """
        Изменить размер изображения, если оно больше max_size. Возвращает путь к новому файлу.
        """
        max_size = max_size or self.max_image_size
        try:
            with Image.open(file_path) as img:
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img = img.copy()
                    img.thumbnail(max_size)
                    out_path = Path(file_path).with_name(f"{Path(file_path).stem}_resized{Path(file_path).suffix}")
                    img.save(out_path)
                    self.logger.info(f"Resized image saved: {out_path}")
                    return str(out_path)
                return file_path
        except Exception as e:
            self.logger.error(f"Failed to resize image: {file_path}, error: {e}")
            return None

    def get_image_dimensions(self, file_path: str) -> Optional[Tuple[int, int]]:
        """Вернуть размеры изображения (width, height)."""
        try:
            with Image.open(file_path) as img:
                return img.size
        except Exception as e:
            self.logger.error(f"Failed to get image dimensions: {file_path}, error: {e}")
            return None

    def compress_video_if_needed(self, file_path: str) -> str:
        """Заглушка: Реализация требует ffmpeg."""
        self.logger.debug("compress_video_if_needed(): Not implemented")
        return file_path

    def get_video_duration(self, file_path: str) -> float:
        """Заглушка: Реализация требует ffprobe."""
        self.logger.debug("get_video_duration(): Not implemented")
        return 0.0

    def create_thumbnail(self, file_path: str) -> Optional[str]:
        """Создать миниатюру для изображения."""
        try:
            with Image.open(file_path) as img:
                thumb_path = str(Path(file_path).with_name(f"thumb_{Path(file_path).name}"))
                img.thumbnail((320, 320))
                img.save(thumb_path)
                self.logger.info(f"Thumbnail created: {thumb_path}")
                return thumb_path
        except Exception as e:
            self.logger.warning(f"Failed to create thumbnail: {file_path}, error: {e}")
            return None

    def get_media_stats(self) -> Dict[str, Any]:
        """Вернуть статистику по медиафайлам (количество, средний размер)."""
        files = self.get_media_files_list()
        stats = {
            "total": len(files),
            "images": 0,
            "videos": 0,
            "documents": 0,
            "avg_size_bytes": 0
        }
        sizes = []
        for file in files:
            typ = self.get_media_type(file)
            stats[typ + "s"] += 1 if typ + "s" in stats else 0
            s = self.get_file_size(file)
            if s > 0:
                sizes.append(s)
        stats["avg_size_bytes"] = int(sum(sizes) / len(sizes)) if sizes else 0
        return stats

    def cleanup_processed_media(self):
        """Очистка временных/обработанных файлов (resized, thumb_ и др.)."""
        for pattern in ["*_resized.*", "thumb_*"]:
            for f in self.media_folder.glob(pattern):
                try:
                    f.unlink()
                    self.logger.info(f"Deleted temp file: {f}")
                except Exception as e:
                    self.logger.warning(f"Can't delete temp file: {f}, error: {e}")
