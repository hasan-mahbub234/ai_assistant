"""
Custom middleware for security, logging, and rate limiting
"""
import time
import logging
from typing import Dict, Any
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.core.config import settings

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses
    """
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        
        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        response.headers["Content-Security-Policy"] = csp
        
        # HSTS for HTTPS
        if settings.ENABLE_HSTS:
            response.headers["Strict-Transport-Security"] = f"max-age={settings.HSTS_MAX_AGE}; includeSubDomains"
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Log all requests and responses
    """
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(
            "Request",
            extra={
                "method": request.method,
                "url": str(request.url),
                "client": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", ""),
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "Request failed",
                extra={
                    "method": request.method,
                    "url": str(request.url),
                    "duration": duration,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise
        
        # Log response
        duration = time.time() - start_time
        logger.info(
            "Response",
            extra={
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "duration": duration,
            }
        )
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Basic rate limiting middleware
    """
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.rate_limit_store: Dict[str, Dict[str, Any]] = {}
    
    async def dispatch(self, request: Request, call_next):
        if not settings.ENABLE_RATE_LIMITING:
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Get or create client data
        client_data = self.rate_limit_store.get(client_ip, {
            "count": 0,
            "window_start": current_time,
        })
        
        # Reset if window expired (1 minute)
        if current_time - client_data["window_start"] > 60:
            client_data = {
                "count": 0,
                "window_start": current_time,
            }
        
        # Check rate limit
        if client_data["count"] >= settings.RATE_LIMIT_PER_MINUTE:
            logger.warning(
                "Rate limit exceeded",
                extra={"client_ip": client_ip, "count": client_data["count"]}
            )
            return Response(
                content="Rate limit exceeded. Please try again later.",
                status_code=429,
                headers={"Retry-After": "60"},
            )
        
        # Increment count
        client_data["count"] += 1
        self.rate_limit_store[client_ip] = client_data
        
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(settings.RATE_LIMIT_PER_MINUTE)
        response.headers["X-RateLimit-Remaining"] = str(
            settings.RATE_LIMIT_PER_MINUTE - client_data["count"]
        )
        response.headers["X-RateLimit-Reset"] = str(
            int(client_data["window_start"] + 60)
        )
        
        return response


def setup_cors(app: ASGIApp) -> ASGIApp:
    """
    Setup CORS middleware
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Accept",
            "Origin",
            "X-Requested-With",
        ],
        expose_headers=["Content-Disposition", "X-RateLimit-*"],
        max_age=600,
    )
    return app
