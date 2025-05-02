from typing import List, TypeVar, Generic
from .contexts import ProcessingContext

T = TypeVar("T", bound=ProcessingContext)


class ProcessingPipeline(Generic[T]):
    """Базовый класс конвейера обработки."""

    def __init__(self):
        """Инициализирует пустой конвейер."""
        self.handlers = []

    def add_handler(self, handler):
        """
        Добавляет обработчик в конвейер.

        Args:
            handler: Обработчик для добавления

        Returns:
            Ссылка на текущий конвейер для цепочки вызовов
        """
        self.handlers.append(handler)
        return self

    def process(self, context: T) -> T:
        """
        Запускает обработку контекста через конвейер.

        Args:
            context: Контекст обработки

        Returns:
            Обработанный контекст
        """
        for handler in self.handlers:
            handler.handle(context)
            # Проверяем на наличие ошибки после каждого обработчика
            if context.has_error:
                # Позволяем некоторым обработчикам выполняться даже при ошибках (например, очистка)
                if (
                    not hasattr(handler, "execute_on_error")
                    or not handler.execute_on_error
                ):
                    break

        return context
