from abc import ABC
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import subprocess
from .models import FaceOperationJobFile, FaceSwapJobRequest, FaceSwapJobResponse


class ProcessingContext(ABC):
    """Базовый абстрактный контекст обработки."""

    def __init__(self):
        self.error: Optional[Exception] = None

    @property
    def has_error(self) -> bool:
        """Проверка наличия ошибки в контексте."""
        return self.error is not None


class FaceVerificationContext(ProcessingContext):
    """Контекст для верификации лица."""

    def __init__(self, image_file: FaceOperationJobFile):
        super().__init__()
        self.image_file = image_file
        self.image_url = image_file.file_url
        self.image_path: Optional[Path] = None
        self.image_data: Optional[np.ndarray] = None
        self.bounding_boxes: Optional[List] = None
        self.face_scores: Optional[List] = None
        self.face_landmarks_5: Optional[List] = None
        self.faces_count: Optional[int] = None
        self.face_classifications: Optional[List[Dict[str, Any]]] = None
        
        # Поле для хранения итогового результата
        self.result: Optional[Dict[str, Any]] = None
        
    @property
    def get_result(self) -> Dict[str, Any]:
        """Формирует результат верификации лица."""
        if self.has_error:
            return {"success": False, "error": str(self.error)}
        
        result = {
            "success": True,
            "faces_count": self.faces_count,
            "faces": []
        }
        
        if self.faces_count and self.face_classifications:
            for i in range(self.faces_count):
                face_data = {
                    "bounding_box": self.bounding_boxes[i].tolist() if hasattr(self.bounding_boxes[i], 'tolist') else self.bounding_boxes[i],
                    "confidence": float(self.face_scores[i]),
                    "classification": self.face_classifications[i]
                }
                result["faces"].append(face_data)
        
        self.result = result
        return result


class FaceSwapContext(ProcessingContext):
    """Контекст для операции замены лица."""

    def __init__(self, job: FaceSwapJobRequest):
        super().__init__()
        self.job = job
        self.source_image_url = job.source_image.file_url
        self.target_image_url = job.target_image.file_url
        self.source_path: Optional[Path] = None
        self.target_path: Optional[Path] = None
        self.output_path: Optional[Path] = None
        self.command: Optional[List[str]] = None
        self.process_result: Optional[subprocess.CompletedProcess] = None
        self.response: Optional[FaceSwapJobResponse] = None
