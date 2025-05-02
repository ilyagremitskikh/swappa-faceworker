from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # Directories
    upload_dir: str
    output_dir: str
    temp_dir: str

    # FastAPI
    service_host: str
    service_port: int
    log_level: str

    #Processing Settings
    max_faces_for_classification: int
    execution_providers: list

    #FaceSwap Configuration
    faceswap_default_processors: list
    faceswap_default_face_detector: str
    faceswap_default_face_selector_mode: str
    
    #Redis Configuration
    redis_host: str
    redis_port: int
    redis_db: int
    redis_url: str
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


config = Config()
