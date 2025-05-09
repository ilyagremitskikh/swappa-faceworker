import logging
import subprocess
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import requests
import numpy as np

from face_operations.utils import parse_extension_from_url
from facefusion import state_manager
from facefusion.face_classifier import classify_face
from facefusion.face_classifier import pre_check as classifier_pre_check
from facefusion.face_detector import detect_faces
from facefusion.face_detector import pre_check as detector_pre_check
from config import config
from .contexts import FaceSwapContext, FaceVerificationContext, ProcessingContext
from .models import FaceOperationJobStatus, FaceSwapJobResponse


logger = logging.getLogger(__name__)


class Handler(ABC):
    """Базовый абстрактный класс обработчика."""

    # Флаг, указывающий, должен ли обработчик выполняться при наличии ошибки
    execute_on_error = False

    @abstractmethod
    def handle(self, context: ProcessingContext) -> None:
        """
        Обработать контекст. Метод должен быть реализован в подклассах.

        Args:
            context: Контекст обработки
        """
        pass


# ==== Обработчики для верификации лица ====


class DownloadImageHandler(Handler):
    """Универсальный обработчик для скачивания изображения."""

    def __init__(self, temp_dir: str, prefix: str = "", field_name: str = None):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.field_name = field_name

    def handle(self, context: ProcessingContext) -> None:
        """Скачивает изображение по URL.

        Args:
            context: Контекст обработки (FaceVerificationContext или FaceSwapContext)
        """
        if context.has_error:
            return

        try:
            # Определяем имена полей в зависимости от типа контекста
            if self.field_name:
                url_field = f"{self.field_name}_image_url"
                path_field = f"{self.field_name}_path"
            elif isinstance(context, FaceVerificationContext):
                url_field = "image_url"
                path_field = "image_path"
            elif isinstance(context, FaceSwapContext):
                # Для обратной совместимости с устаревшим кодом
                if hasattr(context, "source_image_url") and not hasattr(
                    context, "target_path"
                ):
                    url_field = "source_image_url"
                    path_field = "source_path"
                elif hasattr(context, "target_image_url") and not hasattr(
                    context, "source_path"
                ):
                    url_field = "target_image_url"
                    path_field = "target_path"
                else:
                    context.error = Exception(
                        "Неизвестный тип контекста или необходимо указать field_name"
                    )
                    return
            else:
                context.error = Exception("Неизвестный тип контекста")
                return

            # Получаем URL из соответствующего поля контекста
            image_url = getattr(context, url_field)

            # Скачиваем изображение
            response = requests.get(image_url)
            response.raise_for_status()

            # Формируем префикс имени файла если задан
            prefix = f"{self.prefix}_" if self.prefix else ""

            # Сохраняем во временный файл
            image_path = self.temp_dir / f"{prefix}{uuid.uuid4()}.jpg"
            with open(image_path, "wb") as f:
                f.write(response.content)

            # Сохраняем путь в соответствующее поле контекста
            setattr(context, path_field, image_path)
        except Exception as e:
            context.error = Exception(f"Ошибка при скачивании изображения: {e}")


class LoadImageHandler(Handler):
    """Обработчик для загрузки изображения в память."""

    def handle(self, context: FaceVerificationContext) -> None:
        """Загружает изображение из файла в память."""
        if context.has_error:
            return

        try:
            from facefusion.vision import read_static_image

            context.image_data = read_static_image(str(context.image_path))
        except Exception as e:
            context.error = Exception(f"Ошибка при загрузке изображения: {e}")


class DetectFacesHandler(Handler):
    """Обработчик для обнаружения лиц на изображении."""

    def handle(self, context: FaceVerificationContext) -> None:
        """Обнаруживает лица на изображении."""
        if context.has_error:
            return

        try:
            # Инициализация и проверка моделей
            self._initialize_face_detector_state()

            if not detector_pre_check():
                context.error = Exception("Ошибка загрузки моделей детектора лиц")
                return

            # Вызов детектора
            bounding_boxes, face_scores, face_landmarks_5 = detect_faces(
                context.image_data
            )

            # Применяем собственный NMS для удаления дублей
            unique_indices = self._apply_nms(
                bounding_boxes, face_scores, iou_threshold=0.5
            )

            # Отбираем только уникальные обнаружения
            filtered_boxes = [bounding_boxes[i] for i in unique_indices]
            filtered_scores = [face_scores[i] for i in unique_indices]
            filtered_landmarks = [face_landmarks_5[i] for i in unique_indices]

            # Сохраняем результаты
            context.bounding_boxes = filtered_boxes
            context.face_scores = filtered_scores
            context.face_landmarks_5 = filtered_landmarks
            context.faces_count = len(filtered_boxes)

            if context.faces_count == 0:
                context.error = Exception("Лица не обнаружены на изображении")
        except Exception as e:
            context.error = Exception(f"Ошибка при обнаружении лиц: {e}")

    def _initialize_face_detector_state(self):
        """Инициализирует необходимые параметры state_manager для работы детектора лиц."""
        # Установка основных параметров детектора лиц
        if not state_manager.get_item("face_detector_model"):
            state_manager.init_item("face_detector_model", "yolo_face")

        if not state_manager.get_item("face_detector_size"):
            state_manager.init_item("face_detector_size", "640x640")

        if not state_manager.get_item("face_detector_score"):
            state_manager.init_item("face_detector_score", 0.5)

        if not state_manager.get_item("face_detector_angles"):
            state_manager.init_item("face_detector_angles", [0])

        # Установка параметров скачивания
        if not state_manager.get_item("download_providers"):
            state_manager.init_item("download_providers", ["github", "huggingface"])

        if not state_manager.get_item("download_scope"):
            state_manager.init_item("download_scope", "full")

        # Установка параметров выполнения
        if not state_manager.get_item("execution_providers"):
            state_manager.init_item("execution_providers", config.execution_providers)

        if not state_manager.get_item("execution_device_id"):
            state_manager.init_item("execution_device_id", config.execution_device_id)

        # Установка путей для конфигурации
        if not state_manager.get_item("config_path"):
            config_path = str(
                Path(Path(__file__).parent.parent, ".assets", "configs", "default.ini")
            )
            state_manager.init_item("config_path", config_path)

    def _apply_nms(self, boxes, scores, iou_threshold=0.5):
        """Применяет Non-Maximum Suppression для удаления дублирующих обнаружений."""
        if len(boxes) == 0:
            return []

        # Конвертируем в numpy массивы если они не numpy
        if not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)

        # Получаем индексы сортировки по убыванию scores
        indices = np.argsort(scores)[::-1]

        keep_indices = []
        while len(indices) > 0:
            # Добавляем индекс с наивысшим score
            current_index = indices[0]
            keep_indices.append(current_index)

            # Если это последний индекс, выходим
            if len(indices) == 1:
                break

            # Вычисляем IoU между текущим боксом и всеми остальными
            ious = []
            current_box = boxes[current_index]

            for idx in indices[1:]:
                ious.append(self._calculate_iou(current_box, boxes[idx]))

            # Оставляем только индексы с IoU меньше порога
            indices = indices[1:][np.array(ious) < iou_threshold]

        return keep_indices

    def _calculate_iou(self, box1, box2):
        """Вычисляет IoU (Intersection over Union) между двумя боксами."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Вычисляем координаты пересечения
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        # Площадь пересечения
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Площади боксов
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        # IoU = площадь пересечения / площадь объединения
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0


class ClassifyFacesHandler(Handler):
    """Обработчик для классификации лиц на изображении."""

    def handle(self, context: FaceVerificationContext) -> None:
        """Классифицирует лица на изображении."""
        if context.has_error:
            return

        try:
            # Инициализируем необходимые параметры для state_manager
            self._initialize_face_classifier_state()

            # Проверяем и загружаем модели, если необходимо
            if not classifier_pre_check():
                context.error = Exception("Ошибка загрузки моделей классификатора лиц")
                return

            # Проверяем, есть ли обнаруженные лица
            if not hasattr(context, "face_landmarks_5") or not context.face_landmarks_5:
                context.error = Exception("Не найдены ориентиры лиц для классификации")
                return

            # Список для хранения результатов классификации
            context.face_classifications = []

            # Классифицируем каждое лицо
            for landmark in context.face_landmarks_5:
                gender, age, race = classify_face(context.image_data, landmark)
                print(age)
                context.face_classifications.append(
                    {
                        "gender": gender,
                        "age": {"from": age.start, "to": age.stop},
                        "race": race,
                    }
                )

        except Exception as e:
            context.error = Exception(f"Ошибка при классификации лиц: {e}")

    def _initialize_face_classifier_state(self):
        """Инициализирует необходимые параметры state_manager для работы классификатора лиц."""

        # Установка дополнительных параметров
        if not state_manager.get_item("face_selector_mode"):
            state_manager.init_item("face_selector_mode", "many")

        # Установка параметров скачивания
        if not state_manager.get_item("download_providers"):
            state_manager.init_item("download_providers", ["github", "huggingface"])

        if not state_manager.get_item("download_scope"):
            state_manager.init_item("download_scope", "full")

        # Установка параметров выполнения
        if not state_manager.get_item("execution_providers"):
            state_manager.init_item("execution_providers", config.execution_providers)

        if not state_manager.get_item("execution_device_id"):
            state_manager.init_item("execution_device_id", config.execution_device_id)


class ChildProtectionHandler(Handler):
    """Обработчик для защиты детей."""

    def handle(self, context: FaceVerificationContext) -> None:
        """Проверяет наличие несовершеннолетних на изображении."""
        if context.has_error:
            return

        if context.face_classifications:
            for classification in context.face_classifications:
                if classification["age"]["from"] < 18:
                    context.error = Exception("Обнаружены лица несовершеннолетних")


class CleanupHandler(Handler):
    """Обработчик для очистки временных файлов."""

    # Этот обработчик должен выполняться даже при наличии ошибок
    execute_on_error = True

    def handle(self, context: ProcessingContext) -> None:
        """Удаляет временные файлы."""
        if hasattr(context, "image_path") and context.image_path is not None:
            try:
                if context.image_path.exists():
                    context.image_path.unlink()
            except Exception as e:
                logger.warning(f"Не удалось удалить временный файл: {e}")


# ==== Обработчики для FaceSwap ====


class DownloadMediaHandler(Handler):
    """Универсальный обработчик для скачивания изображения."""

    def __init__(self, temp_dir: str, prefix: str = "", field_name: str = None):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.field_name = field_name

    def handle(self, context: ProcessingContext) -> None:
        """Скачивает медиа по URL.

        Args:
            context: Контекст обработки (FaceSwapContext)
        """
        if context.has_error:
            return

        try:
            # Определяем имена полей в зависимости от типа контекста
            if self.field_name:
                url_field = f"{self.field_name}_url"
                path_field = f"{self.field_name}_path"
            elif isinstance(context, FaceSwapContext):
                # Для обратной совместимости с устаревшим кодом
                if hasattr(context, "source_url") and not hasattr(
                    context, "target_path"
                ):
                    url_field = "source_media_url"
                    path_field = "source_media_path"
                elif hasattr(context, "target_url") and not hasattr(
                    context, "source_path"
                ):
                    url_field = "target_url"
                    path_field = "target_path"
                else:
                    context.error = Exception(
                        "Неизвестный тип контекста или необходимо указать field_name"
                    )
                    return
            else:
                context.error = Exception("Неизвестный тип контекста")
                return

            # Получаем URL из соответствующего поля контекста
            media_url = getattr(context, url_field)

            # Скачиваем изображение
            response = requests.get(media_url)
            response.raise_for_status()

            # Формируем префикс имени файла если задан
            prefix = f"{self.prefix}_" if self.prefix else ""

            # Сохраняем во временный файл
            media_path = (
                self.temp_dir
                / f"{prefix}{uuid.uuid4()}.{parse_extension_from_url(media_url)}"
            )
            with open(media_path, "wb") as f:
                f.write(response.content)

            # Сохраняем путь в соответствующее поле контекста
            setattr(context, path_field, media_path)
        except Exception as e:
            context.error = Exception(f"Ошибка при скачивании медиа: {e}")


def DownloadSourceMediaHandler(temp_dir: str) -> DownloadMediaHandler:
    """Создает обработчик для скачивания исходного медиа."""
    return DownloadMediaHandler(temp_dir, prefix="source", field_name="source")


def DownloadTargetMediaHandler(temp_dir: str) -> DownloadMediaHandler:
    """Создает обработчик для скачивания целевого медиа."""
    return DownloadMediaHandler(temp_dir, prefix="target", field_name="target")


class PrepareOutputPathHandler(Handler):
    """Обработчик для подготовки пути выходного файла."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def handle(self, context: FaceSwapContext) -> None:
        """Создает путь для выходного файла."""
        if context.has_error:
            return

        try:
            suffix = context.target_path.suffix
            context.output_path = self.output_dir / f"result_{uuid.uuid4()}{suffix}"
        except Exception as e:
            context.error = Exception(
                f"Ошибка при подготовке пути выходного файла: {e}"
            )


class PrepareCommandHandler(Handler):
    """Обработчик для подготовки команды FaceSwap."""

    def __init__(self, facefusion_script: str, settings_file: str):
        self.facefusion_script = Path(facefusion_script)
        self.settings_file = Path(settings_file)

    def handle(self, context: FaceSwapContext) -> None:
        """Формирует команду для запуска FaceSwap."""
        if context.has_error:
            return

        try:
            # Формирование команды
            context.command = [
                "python",
                str(self.facefusion_script),
                "headless-run",
                "--config-path",
                str(self.settings_file),
                "-s",
                str(context.source_path),
                "-t",
                str(context.target_path),
                "-o",
                str(context.output_path),
                "--face-selector-mode",
                context.job.face_selector_mode.value,
                "--processors",
            ] + [processor.value for processor in context.job.processor]
        except Exception as e:
            context.error = Exception(f"Ошибка при подготовке команды: {e}")


class ExecuteFaceSwapHandler(Handler):
    """Обработчик для выполнения FaceSwap."""

    def handle(self, context: FaceSwapContext) -> None:
        """Выполняет команду FaceSwap."""
        if context.has_error:
            return

        try:
            logger.info(f"Запуск команды FaceSwap: {' '.join(context.command)}")
            process = subprocess.run(
                context.command, capture_output=True, text=True, check=False
            )

            logger.debug(f"FaceSwap Stderr: {process.stderr}")
            logger.debug(f"FaceSwap Stdout: {process.stdout}")

            context.process_result = process
        except Exception as e:
            context.error = Exception(f"Ошибка при выполнении FaceSwap: {e}")


class PrepareResponseHandler(Handler):
    """Обработчик для подготовки ответа FaceSwap."""

    # Этот обработчик должен выполняться даже при наличии ошибок
    execute_on_error = True

    def handle(self, context: FaceSwapContext) -> None:
        """Подготавливает ответ с результатом операции."""
        try:
            if context.has_error:
                context.response = FaceSwapJobResponse(
                    id=context.job.id,
                    status=FaceOperationJobStatus.FAILED,
                    output_path="",
                )
                return

            if context.process_result.returncode == 0:
                context.response = FaceSwapJobResponse(
                    id=context.job.id,
                    status=FaceOperationJobStatus.COMPLETED,
                    output_path=str(context.output_path),
                )
            else:
                logger.error(
                    f"Ошибка выполнения FaceFusion FaceSwap (код {context.process_result.returncode})\n"
                    f"Stderr: {context.process_result.stderr}\n"
                    f"Stdout: {context.process_result.stdout}"
                )
                context.response = FaceSwapJobResponse(
                    id=context.job.id,
                    status=FaceOperationJobStatus.FAILED,
                    output_path="",
                )
        except Exception as e:
            context.error = Exception(f"Ошибка при подготовке ответа: {e}")
            context.response = FaceSwapJobResponse(
                id=context.job.id, status=FaceOperationJobStatus.FAILED, output_path=""
            )


class CleanupFaceSwapHandler(Handler):
    """Обработчик для очистки временных файлов FaceSwap."""

    # Этот обработчик должен выполняться даже при наличии ошибок
    execute_on_error = True

    def handle(self, context: FaceSwapContext) -> None:
        """Удаляет временные файлы."""
        try:
            if context.source_path and context.source_path.exists():
                context.source_path.unlink()
        except Exception as e:
            logger.warning(f"Не удалось удалить исходный файл: {e}")

        try:
            if context.target_path and context.target_path.exists():
                context.target_path.unlink()
        except Exception as e:
            logger.warning(f"Не удалось удалить целевой файл: {e}")
