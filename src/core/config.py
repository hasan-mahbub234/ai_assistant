"""
Application configuration using Pydantic Settings
"""
import secrets
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, PostgresDsn, validator


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    PROJECT_NAME: str = "Legal SaaS API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    JWT_SECRET: str = secrets.token_urlsafe(32)
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 12
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    PASSWORD_RESET_TOKEN_EXPIRE_MINUTES: int = 15
    
    # Database
    DATABASE_URL: str = "postgresql://mahbubdb:mahbub123@localhost:5432/legal_saas"
    
    # CORS
    CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",
        "http://localhost:8080",
    ]
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str] | List[AnyHttpUrl]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        return []
    
    # Allowed Hosts
    ALLOWED_HOSTS: List[str] = [
        "localhost",
        "127.0.0.1",
        "*.railway.app",
    ]
    
    # File Upload
    UPLOAD_DIR: str = "/app/uploads"
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_FILE_TYPES: List[str] = [".pdf", ".docx", ".txt"]
    
    # AI/ML Services
    GEMINI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"  # or "mixtral-8x7b-32768", "gemma2-9b-it"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX_NAME: str = "legal-documents"
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    HUGGINGFACE_ACCESS_TOKEN: Optional[str] = None
    
    # Redis (for caching/rate limiting)
    REDIS_URL: Optional[str] = "redis://localhost:6379/0"
    
    # Rate Limiting
    ENABLE_RATE_LIMITING: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Email
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAIL_FROM: str = "noreply@legalsaas.com"
    
    # Monitoring
    SENTRY_DSN: Optional[str] = None
    LOG_LEVEL: str = "INFO"
    
    # Security Headers
    ENABLE_HSTS: bool = True
    HSTS_MAX_AGE: int = 31536000  # 1 year
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()