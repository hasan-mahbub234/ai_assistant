from pinecone import Pinecone, ServerlessSpec
import logging
import time
from src.config import config

logger = logging.getLogger(__name__)


class PineconeManager:
    """Manage Pinecone operations"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize Pinecone connection"""
        logger.info("Initializing Pinecone...")
        
        if not config.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not configured")
        
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index_name = config.PINECONE_INDEX_NAME
        self._setup_index()
    
    def _setup_index(self):
        """Setup or verify Pinecone index exists"""
        try:
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes.indexes]
            
            if self.index_name not in index_names:
                logger.info(f"Creating index: {self.index_name} (dim={config.EMBED_DIMENSION})")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=config.EMBED_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                logger.info("Waiting for index creation...")
                time.sleep(15)
            
            self.index = self.pc.Index(self.index_name)
            stats = self.index.describe_index_stats()
            logger.info(f"Pinecone connected. Total vectors: {stats.total_vector_count}")
            
        except Exception as e:
            logger.error(f"Failed to setup Pinecone index: {e}")
            raise
    
    def search(self, query_embedding, top_k=5, include_metadata=True):
        """Search vectors"""
        return self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=include_metadata
        )
    
    def upsert(self, vectors):
        """Upsert vectors"""
        self.index.upsert(vectors=vectors)
    
    def fetch(self, vector_ids):
        """Fetch vectors by IDs"""
        return self.index.fetch(ids=vector_ids)
    
    def get_stats(self):
        """Get index statistics"""
        return self.index.describe_index_stats()
    
    def check_connection(self):
        """Check Pinecone connection status"""
        try:
            stats = self.get_stats()
            return {
                "status": "connected",
                "message": "Pinecone working",
                "stats": {
                    "total_vector_count": stats.total_vector_count,
                    "dimension": stats.dimension
                }
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}