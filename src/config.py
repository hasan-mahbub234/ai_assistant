from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database
    DB_HOST: str
    DB_PORT: int = 3306
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    
    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "ecommerce-customer-support"
    PINECONE_ENVIRONMENT: Optional[str] = "us-east-1"
    
    # Groq
    GROQ_API_KEY: str
    
    # Hugging Face (optional)
    HF_API_TOKEN: Optional[str] = None
    
    # Gemini (optional)
    GEMINI_API_KEY: Optional[str] = None
    
    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    EMBED_DIMENSION: int = 768
    
    # App
    LOG_LEVEL: str = "INFO"
    PORT: int = 8000
    
    # Vectorization
    PINECONE_BATCH_SIZE: int = 50
    UPSERT_SLEEP: float = 0.15
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # This will ignore any extra fields in .env


config = Settings()