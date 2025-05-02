import logging
import os
from typing import Any, Dict

from .contexts import FaceSwapContext, FaceVerificationContext
from .handlers import (
    ClassifyFacesHandler,
    CleanupFaceSwapHandler,
    CleanupHandler,
    ChildProtectionHandler,
    DetectFacesHandler,
    DownloadImageHandler,
    ExecuteFaceSwapHandler,
    LoadImageHandler,
    PrepareCommandHandler,
    PrepareOutputPathHandler,
    PrepareResponseHandler,
)
from .models import FaceOperationJobFile, FaceSwapJobRequest, FaceSwapJobResponse
from .pipeline import ProcessingPipeline

logger = logging.getLogger(__name__)

# Константы для конфигурации
#TODO: Перенести в конфиг
TEMP_DIR = os.environ.get("TEMP_DIR", "/workspaces/swappa/worker/tmp/facefusion")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspaces/swappa/worker/tmp/facefusion/output")
FACEFUSION_SCRIPT = os.environ.get("FACEFUSION_SCRIPT", "/workspaces/swappa/worker/facefusion.py")
SETTINGS_FILE = os.environ.get("SETTINGS_FILE", "/workspaces/swappa/worker/facefusion.ini")


class FaceOperationsService:
    """Сервис для выполнения операций с лицами."""

    @staticmethod
    def run_face_verification(image: FaceOperationJobFile) -> Dict[str, Any]:
        """
        Верифицирует лицо на изображении.

        Args:
            image: Файл с изображением лица

        Returns:
            Результат верификации с классификацией лица

        Raises:
            Exception: При ошибке в процессе верификации
        """
        logger.info(f"Запуск верификации лица: {image.file_url}")

        try:
            # Создаем конвейер обработки
            pipeline = ProcessingPipeline()
            # Используем универсальный обработчик без дополнительных параметров
            pipeline.add_handler(DownloadImageHandler(TEMP_DIR))
            pipeline.add_handler(LoadImageHandler())
            pipeline.add_handler(DetectFacesHandler())
            pipeline.add_handler(ClassifyFacesHandler())
            pipeline.add_handler(ChildProtectionHandler())
            pipeline.add_handler(CleanupHandler())

            # Создаем контекст
            context = FaceVerificationContext(image)

            # Запускаем обработку
            processed_context = pipeline.process(context)

            # Проверяем на наличие ошибок
            if processed_context.has_error:
                raise processed_context.error

            # Получаем и возвращаем результат
            result = processed_context.get_result
            
            logger.info(
                f"Верификация лица успешно завершена: {result}"
            )
            return result

        except Exception as e:
            logger.error(f"Ошибка при верификации лица: {e}", exc_info=True)
            raise Exception(f"Ошибка при верификации лица: {e}")

    @staticmethod
    def run_face_swap(job: FaceSwapJobRequest) -> FaceSwapJobResponse:
        """
        Запускает процесс FaceFusion FaceSwap.

        Args:
            job: Запрос на выполнение FaceSwap

        Returns:
            Ответ с результатом выполнения операции
        """
        logger.info(f"Запуск FaceSwap: {job.id}")

        try:
            # Создаем конвейер обработки
            pipeline = ProcessingPipeline()
            # Используем универсальный обработчик с указанием поля
            pipeline.add_handler(DownloadImageHandler(TEMP_DIR, prefix="source", field_name="source"))
            pipeline.add_handler(DownloadImageHandler(TEMP_DIR, prefix="target", field_name="target"))
            pipeline.add_handler(PrepareOutputPathHandler(OUTPUT_DIR))
            pipeline.add_handler(
                PrepareCommandHandler(FACEFUSION_SCRIPT, SETTINGS_FILE)
            )
            pipeline.add_handler(ExecuteFaceSwapHandler())
            pipeline.add_handler(PrepareResponseHandler())
            pipeline.add_handler(CleanupFaceSwapHandler())

            # Создаем контекст
            context = FaceSwapContext(job)

            # Запускаем обработку
            processed_context = pipeline.process(context)

            if processed_context.has_error:
                raise processed_context.error

            # Возвращаем ответ
            response = processed_context.response
            logger.info(f"FaceSwap завершен: {response.status}")
            return response

        except Exception as e:
            logger.error(
                f"Неожиданная ошибка при выполнении FaceSwap: {e}", exc_info=True
            )
            return FaceSwapJobResponse(id=job.id, status="FAILED", output_path="")
