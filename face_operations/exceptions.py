from enum import Enum
from typing import Optional, Dict, Any


class ErrorCode(Enum):
    DOWNLOAD_ERROR = "DOWNLOAD_ERROR"
    IMAGE_PROCESSING_ERROR = "IMAGE_PROCESSING_ERROR"
    FACE_DETECTION_ERROR = "FACE_DETECTION_ERROR"
    FACE_CLASSIFICATION_ERROR = "FACE_CLASSIFICATION_ERROR"
    CHILD_PROTECTION_ERROR = "CHILD_PROTECTION_ERROR"
    FACE_SWAP_ERROR = "FACE_SWAP_ERROR"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class OperationError(Exception):
    """Базовый класс для ошибок операций с лицами"""
    
    def __init__(
        self, 
        message: str, 
        code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)