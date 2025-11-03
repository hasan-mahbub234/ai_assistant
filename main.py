from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from src.ai.routes import ai_router
from src.ai.ai import EcommerceAICustomerSupport

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ai_support = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for Vercel"""
    global ai_support
    logger.info("🚀 AI Customer Support Service Starting...")
    
    try:
        # Initialize AI system (NO AUTOMATIC VECTORIZATION)
        ai_support = EcommerceAICustomerSupport()
        
        # Check Pinecone status only
        pinecone_status = ai_support.check_pinecone_status()
        logger.info(f"📚 Pinecone Status: {pinecone_status}")
        
        if pinecone_status['status'] == 'connected':
            vector_count = pinecone_status['stats']['total_vector_count']
            if vector_count > 0:
                logger.info(f"✅ Pinecone connected with {vector_count} vectors")
            else:
                logger.warning("⚠️ Pinecone connected but has 0 vectors")
                logger.info("💡 Run: python vectorize_database.py to populate vectors")
        else:
            logger.warning(f"⚠️ Pinecone not available: {pinecone_status['message']}")
        
        logger.info("✅ AI Service Ready!")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        import traceback
        traceback.print_exc()
    
    yield
    
    logger.info("🛑 AI Service Shutting Down...")

app = FastAPI(
    title="Ecommerce AI Customer Support",
    description="Multilingual AI Customer Support with Product Search & Order Tracking",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Simplified for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include AI router
app.include_router(ai_router, prefix="/api/v1/ai", tags=["AI"])

@app.get("/")
async def root():
    return {"message": "Ecommerce AI Customer Support API is running"}

@app.get("/health")
async def health_check():
    global ai_support
    pinecone_status = ai_support.check_pinecone_status() if ai_support else {"status": "unknown"}
    return {
        "status": "healthy", 
        "service": "AI Customer Support", 
        "pinecone": pinecone_status
    }

handler = app