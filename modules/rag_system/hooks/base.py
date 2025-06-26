class BaseHook:
    """
    Базовый интерфейс для всех хуков.
    """
    params = {}
    conflicts = set()

    def __call__(self, text: str, meta: dict, **context):
        """
        Основной интерфейс хука. Должен быть переопределён в наследниках.
        """
        raise NotImplementedError("Метод __call__ должен быть реализован в наследнике.")

    def is_idempotent(self, old_text: str, new_text: str) -> bool:
        """
        Определяет идемпотентность применения хука.
        """
        return old_text == new_text

    def summary(self, old_text: str, new_text: str) -> str:
        """
        Кратко описывает изменения.
        """
        return f"Changed {abs(len(new_text) - len(old_text))} chars"

    @classmethod
    def get_conflicts(cls):
        """
        Возвращает множество имен конфликтующих хуков.
        """
        return getattr(cls, "conflicts", set())
