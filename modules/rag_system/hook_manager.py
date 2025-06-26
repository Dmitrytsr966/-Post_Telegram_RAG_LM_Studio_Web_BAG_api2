import logging
from typing import Callable, List, Dict, Optional, Any, Tuple, Set, Type

class HookConflictError(Exception):
    """Custom exception for hook conflicts."""
    pass

class HookResult:
    """
    Результат применения одного хука:
    - old_text, new_text
    - изменено символов/слов
    - параметры хука
    - ошибки (если были)
    - idempotency
    """
    def __init__(self, hook_name: str, old_text: str, new_text: str, params: dict = None,
                 error: Optional[str] = None, idempotent: Optional[bool] = None, summary: Optional[str] = None):
        self.hook_name = hook_name
        self.old_text = old_text
        self.new_text = new_text
        self.params = params or {}
        self.error = error
        self.idempotent = idempotent
        self.summary = summary
        self.chars_changed = abs(len(new_text) - len(old_text))
        self.words_changed = abs(len(new_text.split()) - len(old_text.split()))

    def as_dict(self) -> dict:
        return {
            "hook": self.hook_name,
            "chars_changed": self.chars_changed,
            "words_changed": self.words_changed,
            "params": self.params,
            "error": self.error,
            "idempotent": self.idempotent,
            "summary": self.summary
        }

class BaseHook:
    """
    Базовый интерфейс для хуков.
    Каждый хук должен реализовать __call__ и может реализовать conflicts, is_idempotent, summary.
    """
    params: dict = {}
    conflicts: Set[str] = set()

    def __call__(self, text: str, meta: dict, **context) -> str:
        raise NotImplementedError

    def is_idempotent(self, old_text: str, new_text: str) -> bool:
        """Определяет, является ли применение хука идемпотентным (одинаковый результат при повторе)."""
        return old_text == new_text

    def summary(self, old_text: str, new_text: str) -> str:
        """Возвращает краткое описание изменений (для аналитики/мониторинга)."""
        return f"Changed {abs(len(new_text) - len(old_text))} chars"

    @classmethod
    def get_conflicts(cls) -> Set[str]:
        """Возвращает имена конфликтующих хуков."""
        return getattr(cls, "conflicts", set())

class HookManager:
    """
    Управляет регистрацией и применением хуков (pre/post, по формату и глобально).
    Каждый хук обязан быть callable: (text, meta, **context) -> str.
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("HookManager")
        self.pre_hooks: Dict[Optional[str], List[Callable]] = {}
        self.post_hooks: Dict[Optional[str], List[Callable]] = {}

    # --- Registration Methods ---

    def register_pre_hook(self, hook: Callable, formats: Optional[Any] = None):
        self._register_hook(hook, self.pre_hooks, formats, hook_type='pre')

    def register_post_hook(self, hook: Callable, formats: Optional[Any] = None):
        self._register_hook(hook, self.post_hooks, formats, hook_type='post')

    def remove_pre_hook(self, hook: Callable, formats: Optional[Any] = None):
        self._remove_hook(hook, self.pre_hooks, formats)

    def remove_post_hook(self, hook: Callable, formats: Optional[Any] = None):
        self._remove_hook(hook, self.post_hooks, formats)

    def _register_hook(self, hook: Callable, hooks_dict: Dict[Optional[str], List[Callable]],
                       formats: Optional[Any], hook_type: str = 'pre'):
        if formats is None:
            formats = [None]
        elif isinstance(formats, str):
            formats = [formats]

        hook_class = hook.__class__ if not isinstance(hook, type) else hook
        hook_name = hook_class.__name__

        # Проверка конфликтов среди уже зарегистрированных хуков
        for fmt in formats:
            hook_conflicts = set()
            if hasattr(hook, "get_conflicts"):
                hook_conflicts = hook.get_conflicts()
            for registered in hooks_dict.get(fmt, []):
                reg_name = registered.__class__.__name__
                if reg_name in hook_conflicts:
                    msg = f"Conflict: {hook_name} conflicts with already registered {reg_name} for format '{fmt}'"
                    self.logger.error(msg)
                    raise HookConflictError(msg)
                # Взаимная проверка: если рег-хук конфликтует с новым
                if hasattr(registered, "get_conflicts"):
                    if hook_name in registered.get_conflicts():
                        msg = f"Conflict: {reg_name} conflicts with registering {hook_name} for format '{fmt}'"
                        self.logger.error(msg)
                        raise HookConflictError(msg)
            hooks_dict.setdefault(fmt, []).append(hook)
            self.logger.info(f"Registered {hook_type}-hook '{hook_name}' for format '{fmt}'")

    def _remove_hook(self, hook: Callable, hooks_dict: Dict[Optional[str], List[Callable]], formats: Optional[Any]):
        if formats is None:
            formats = [None]
        elif isinstance(formats, str):
            formats = [formats]
        for fmt in formats:
            if fmt in hooks_dict:
                hooks_dict[fmt] = [h for h in hooks_dict[fmt] if h != hook]
                self.logger.info(f"Removed hook '{hook.__class__.__name__}' from format '{fmt}'")

    # --- Hook Application Methods ---

    def apply_hooks(
        self,
        text: str,
        meta: dict,
        fmt: str,
        hooks: Dict[Optional[str], List[Callable]],
        context: Optional[dict] = None
    ) -> Tuple[str, List[dict]]:
        """
        Применяет цепочку хуков (pre/post) к тексту.
        Возвращает новый текст и список dict-результатов по каждому хук-вызову.
        Не останавливает цепочку при ошибке одного из хуков.
        """
        context = context or {}
        hook_results: List[dict] = []
        # Собираем цепочку: сначала по формату, затем глобальные
        chain = hooks.get(fmt, []) + hooks.get(None, [])
        for hook in chain:
            hook_class = hook.__class__ if not isinstance(hook, type) else hook
            hook_name = hook_class.__name__
            old_text = text
            params = getattr(hook, "params", {}) if hasattr(hook, "params") else {}
            idempotent = None
            summary = None
            error = None
            try:
                # Передаем формат/расширение как context, если нужно
                new_text = hook(text, meta, **context)
                # Проверка idempotency (если реализовано)
                if hasattr(hook, "is_idempotent"):
                    idempotent = hook.is_idempotent(old_text, new_text)
                # Получить summary, если есть
                if hasattr(hook, "summary"):
                    summary = hook.summary(old_text, new_text)
            except Exception as e:
                error = str(e)
                self.logger.error(f"Hook '{hook_name}' failed: {e}", exc_info=True)
                new_text = text  # Fail-safe: текст не меняем
            result = HookResult(
                hook_name=hook_name,
                old_text=old_text,
                new_text=new_text,
                params=params,
                error=error,
                idempotent=idempotent,
                summary=summary
            )
            hook_results.append(result.as_dict())
            text = new_text  # Продолжаем цепочку
        return text, hook_results

    def apply_pre_hooks(self, text: str, meta: dict, fmt: str, context: Optional[dict] = None) -> Tuple[str, List[dict]]:
        return self.apply_hooks(text, meta, fmt, self.pre_hooks, context)

    def apply_post_hooks(self, text: str, meta: dict, fmt: str, context: Optional[dict] = None) -> Tuple[str, List[dict]]:
        return self.apply_hooks(text, meta, fmt, self.post_hooks, context)

    # --- Информационные методы ---

    def get_registered_hooks(self, hook_type: str = 'pre') -> Dict[Optional[str], List[str]]:
        """
        Возвращает список зарегистрированных хуков (по типу и формату).
        """
        hooks_dict = self.pre_hooks if hook_type=='pre' else self.post_hooks
        out = {}
        for fmt, hook_list in hooks_dict.items():
            out[fmt] = [h.__class__.__name__ for h in hook_list]
        return out

    def check_all_conflicts(self) -> List[str]:
        """
        Проверяет наличие конфликтов среди всех зарегистрированных хуков.
        Возвращает список строк с описанием конфликтов.
        """
        conflicts = []
        for hooks_dict, typ in [(self.pre_hooks, "pre"), (self.post_hooks, "post")]:
            for fmt, hook_list in hooks_dict.items():
                names = [h.__class__.__name__ for h in hook_list]
                for idx, hook in enumerate(hook_list):
                    hook_name = hook.__class__.__name__
                    hook_conflicts = set()
                    if hasattr(hook, "get_conflicts"):
                        hook_conflicts = hook.get_conflicts()
                    for other in hook_list[idx+1:]:
                        other_name = other.__class__.__name__
                        # Взаимная проверка
                        if other_name in hook_conflicts:
                            msg = f"{typ}-hook conflict: {hook_name} conflicts with {other_name} for format '{fmt}'"
                            conflicts.append(msg)
                        if hasattr(other, "get_conflicts") and hook_name in other.get_conflicts():
                            msg = f"{typ}-hook conflict: {other_name} conflicts with {hook_name} for format '{fmt}'"
                            conflicts.append(msg)
        return conflicts
