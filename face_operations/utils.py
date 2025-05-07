import uuid
from pathlib import Path


def parse_extension_from_url(url: str) -> str:
    """
    Парсит расширение из URL.
    """
    return Path(url).suffix


def generate_output_path(output_dir: Path, suffix: str = ".jpg") -> Path:
    """
    Генерирует путь для выходного файла.

    Args:
        output_dir: Директория для выходных файлов
        suffix: Расширение файла

    Returns:
        Путь к выходному файлу
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{uuid.uuid4()}{suffix}"


def ensure_dir_exists(path: Path) -> Path:
    """
    Убеждается, что директория существует.

    Args:
        path: Путь к директории

    Returns:
        Тот же путь для цепочки вызовов
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
