"""
Database configuration and session management
"""
import logging
from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

from src.core.config import settings

logger = logging.getLogger(__name__)

# Create engine with connection pooling disabled for serverless (Railway)
engine = create_engine(
    str(settings.DATABASE_URL or "sqlite:///./local.db"),
    poolclass=NullPool if str(settings.DATABASE_URL).startswith("postgres") else None,
    echo=settings.ENVIRONMENT == "development",
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args=(
        {"connect_timeout": 10, "application_name": "legal-saas-api"}
        if str(settings.DATABASE_URL).startswith("postgres")
        else {"check_same_thread": False}
    )
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
    class_=Session,
)

# Base class for models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for getting database session
    """
    db: Optional[Session] = None
    try:
        db = SessionLocal()
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database session error: {str(e)}")
        raise
    finally:
        if db:
            db.close()


@contextmanager
def db_session():
    """
    Context manager for database sessions
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def check_db_connection() -> bool:
    """
    Check database connection health
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {str(e)}")
        return False


def create_tables():
    """
    Create all tables (for development only)
    """
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def drop_tables():
    """
    Drop all tables (for testing only)
    """
    Base.metadata.drop_all(bind=engine)
    logger.info("Database tables dropped successfully")
