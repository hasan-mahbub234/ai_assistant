from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manage embeddings with support for multiple models"""
    
    # Better embedding models optimized for semantic search
    MODELS = {
        'default': 'sentence-transformers/all-mpnet-base-v2',  # Better for general search
        'semantic': 'sentence-transformers/all-MiniLM-L6-v2',  # Lightweight
        'multilingual': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',  # Better for Bengali/Banglish
        'product': 'sentence-transformers/all-MiniLM-L6-v2',  # For product search
    }
    
    _instances = {}
    
    @staticmethod
    def get_embeddings(model_type='multilingual'):
        """Get embedding model (cached)"""
        if model_type not in EmbeddingManager._instances:
            model_name = EmbeddingManager.MODELS.get(model_type, EmbeddingManager.MODELS['default'])
            logger.info(f"Loading embedding model: {model_name}")
            
            EmbeddingManager._instances[model_type] = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        return EmbeddingManager._instances[model_type]
    
    @staticmethod
    def embed_text(text, model_type='multilingual'):
        """Embed a single text"""
        embeddings = EmbeddingManager.get_embeddings(model_type)
        return embeddings.embed_query(text)
    
    @staticmethod
    def embed_documents(texts, model_type='multilingual'):
        """Embed multiple documents"""
        embeddings = EmbeddingManager.get_embeddings(model_type)
        return embeddings.embed_documents(texts)
