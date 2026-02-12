# config.py - Centralized configuration management
import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Supabase
    SUPABASE_URL: str
    SUPABASE_SECRET: str
    SUPABASE_BUCKET_NAME: str = "product_image"
    
    # JWT
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Email (Optional)
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    FROM_EMAIL: str = "noreply@civiclink.com"
    
    # Application
    APP_NAME: str = "CivicLink"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = True
    ALLOWED_ORIGINS: str = "*"
    
    # ML Model
    YOLO_MODEL_PATH: str = "models/pothole_yolov8_best.pt"
    MIN_CONFIDENCE: float = 0.1
    
    # Geospatial
    NEARBY_RADIUS_KM: float = 0.5
    OSM_SEARCH_RADIUS_M: int = 500
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    @property
    def allowed_origins_list(self) -> List[str]:
        if self.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]

# Global settings instance
settings = Settings()
