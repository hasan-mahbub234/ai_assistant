"""Legal SaaS - Backend Package"""

__version__ = "1.0.0"

from src.auth import auth_router
from src.documents import documents_router
from src.ai import ai_router

__all__ = ["auth_router", "documents_router", "ai_router"]