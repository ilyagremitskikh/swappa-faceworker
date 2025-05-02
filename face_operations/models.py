from dataclasses import dataclass
from enum import Enum
from typing import List


@dataclass
class FaceOperationJobFile:
    """Файл для задания FaceFusion."""

    file_url: str


class FaceOperationJobStatus(str, Enum):
    """Статус операции с лицом."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PROCESSING = "PROCESSING"


class ProcessorType(str, Enum):
    """Типы процессоров для FaceSwap."""

    FACE_SWAPPER = "face_swapper"
    FACE_ENHANCER = "face_enhancer"
    FRAME_ENHANCER = "frame_enhancer"


class FaceSelectorMode(str, Enum):
    """Режимы выбора лица."""

    ONE_FACE = "one"
    MANY_FACES = "many"


@dataclass
class FaceDetectionJobRequest:
    """Запрос на выполнение FaceDetection."""
    image: FaceOperationJobFile


@dataclass
class FaceDetectionJobResponse:
    """Ответ на выполнение FaceDetection."""

    success: bool
    faces_count: int
    faces: list
    confidence: float
    classification: dict


@dataclass
class FaceSwapJobRequest:
    """Запрос на выполнение FaceSwap."""

    id: str
    source_image: FaceOperationJobFile
    target_image: FaceOperationJobFile
    face_selector_mode: FaceSelectorMode
    processor: List[ProcessorType]


@dataclass
class FaceSwapJobResponse:
    """Ответ на выполнение FaceSwap."""

    id: str
    status: FaceOperationJobStatus
    output_path: str
