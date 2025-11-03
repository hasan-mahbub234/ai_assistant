# config.py
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # MySQL Database configuration
    DB_HOST: str 
    DB_PORT: int 
    DB_NAME: str 
    DB_USER: str 
    DB_PASSWORD: str 
    
    # AI Service APIs
    GROQ_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str 
    PINECONE_INDEX_NAME: str 
    
    # Optional: Hugging Face
    HF_API_TOKEN: str 
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global configuration
config = Settings()