"""
Utility functions and helpers
"""
import json
import logging
import structlog
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import uuid4
from fastapi.encoders import jsonable_encoder

from src.core.config import settings


def setup_logging() -> logging.Logger:
    """
    Setup structured logging with structlog and standard logging.
    Ensures 'logs/' directory exists.
    """
    # Ensure logs directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.LOG_LEVEL)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "app.log")),
            logging.StreamHandler(),
        ],
    )

    return structlog.get_logger()


def generate_uuid() -> str:
    """Generate a random UUID string."""
    return str(uuid4())


def utc_now() -> datetime:
    """Get the current UTC datetime."""
    return datetime.now(timezone.utc)


def format_datetime(dt: datetime) -> str:
    """Format datetime to ISO 8601 string."""
    return dt.isoformat()


def safe_json_dumps(data: Any) -> str:
    """Safely serialize data to JSON."""
    return json.dumps(jsonable_encoder(data))


def mask_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mask sensitive fields in a dictionary for logging.
    Fields masked: password, token, secret, api_key, authorization
    """
    masked = data.copy()
    sensitive_fields = ["password", "token", "secret", "api_key", "authorization"]

    for key in masked.keys():
        if any(sensitive in key.lower() for sensitive in sensitive_fields):
            masked[key] = "***MASKED***"

    return masked


def validate_email(email: str) -> bool:
    """Validate email format using regex."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def get_file_extension(filename: str) -> str:
    """Return the file extension in lowercase, empty string if none."""
    return filename.rsplit('.', 1)[-1].lower() if '.' in filename else ""


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"
