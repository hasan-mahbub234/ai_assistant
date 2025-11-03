import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.ai.routes import ai_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

version = "v1"

ai_support_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown events for AI service only."""
    logger.info("🚀 AI Service is starting...")
    
    global ai_support_instance
    
    # Initialize AI system (ONLY ONCE via singleton)
    try:
        from src.ai.ai import EcommerceAICustomerSupport
        ai_support_instance = EcommerceAICustomerSupport()
        logger.info("✅ AI system initialized successfully (singleton pattern)")
        
        # Check Pinecone status
        pinecone_status = ai_support_instance.check_pinecone_status()
        logger.info(f"📊 Pinecone Status: {pinecone_status}")
        
    except Exception as e:
        logger.error(f"❌ AI system initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    yield
    
    logger.info("🛑 AI Service is shutting down...")

def create_app() -> FastAPI:
    """Factory function to create and configure the AI-only FastAPI application."""
    app = FastAPI(
        title="E-commerce AI Service",
        description="AI Customer Support & Product Recommendation API",
        version=version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Configure CORS for Vercel
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:3001",
            "http://192.168.0.100:3000",
            "http://192.168.0.101:3000",
            "http://192.168.0.102:3000",
            "http://192.168.0.103:3000",
            "http://192.168.114.253:3000",
            "https://toprateddesigner.com",
            "https://ecommerce-admin-panel-next-js.vercel.app",
            "https://shulovmall.com",
            "https://www.shulovmall.com",
            "https://*.vercel.app"
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error"},
        )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger.info(f"AI Request: {request.method} {request.url}")
        try:
            response = await call_next(request)
            logger.info(f"AI Response status: {response.status_code}")
            return response
        except Exception as e:
            logger.error(f"AI Request failed: {str(e)}", exc_info=True)
            raise

    # Include ONLY AI router
    routers = [
        (ai_router, "ai"),
    ]

    for router, tag in routers:
        try:
            app.include_router(
                router,
                prefix=f"/api/{version}/{tag}",
                tags=[tag]
            )
            logger.info(f"✅ Successfully added {tag} router")
        except Exception as e:
            logger.error(f"❌ Failed to add {tag} router: {str(e)}")
            raise

    # Health check endpoint
    @app.get("/health", tags=["monitoring"])
    async def health_check():
        return {
            "status": "healthy", 
            "service": "AI Customer Support",
            "deployment": "vercel"
        }
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "E-commerce AI Customer Support API",
            "version": version,
            "docs": "/docs"
        }

    return app

# Create the application instance
app = create_app()

# Vercel requires this for serverless functions
handler = app
