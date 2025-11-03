# config.py
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # MySQL Database configuration
    DB_HOST: str = "43.204.142.93"
    DB_PORT: int = 3306
    DB_NAME: str = "ecommerce_website"
    DB_USER: str = "mahbubdb"
    DB_PASSWORD: str = "mahbub123"
    
    # AI Service APIs
    GROQ_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str = "us-east-1"  # Add this
    PINECONE_INDEX_NAME: str = "ecommerce-customer-support"  # Add this
    
    # Optional: Hugging Face
    HF_API_TOKEN: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global configuration
config = Settings()