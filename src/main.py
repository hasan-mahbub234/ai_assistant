"""FastAPI application main entry point"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.database import engine, Base
from src.auth import auth_router
from src.documents import documents_router
from src.ai import ai_router

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Legal SaaS API",
    description="AI-powered legal document analysis platform with RAG and Gemini integration",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include routers with prefixes
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(ai_router)

@app.get("/", tags=["Health"])
def root():
    """Root endpoint"""
    return {
        "message": "Legal SaaS API is running",
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": [
            "/api/v1/auth",
            "/api/v1/documents",
            "/api/v1/ai",
            "/docs",
            "/health"
        ]
    }

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Legal SaaS API",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)