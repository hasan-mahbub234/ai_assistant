import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manage embeddings with support for multiple models"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize embedding model"""
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
        logger.info(f"Loading embedding model: {model_name}")
        
        try:
            # Try to use the new langchain-huggingface package
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                logger.info("Using langchain-huggingface package")
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                logger.info("Using langchain_community.embeddings (legacy)")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'normalize_embeddings': True,
                }
            )
            
            # Test embedding
            test_embedding = self.embeddings.embed_query("test")
            logger.info(f"Embeddings initialized (dimension: {len(test_embedding)})")
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def embed_query(self, text: str):
        """Embed a single query"""
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: list):
        """Embed multiple documents"""
        return self.embeddings.embed_documents(texts)
    
    def get_dimension(self):
        """Get embedding dimension"""
        return int(os.getenv("EMBED_DIMENSION", 768))