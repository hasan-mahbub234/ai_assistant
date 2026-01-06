import pymysql
from pymysql.cursors import DictCursor
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Database connection manager"""
    
    def get_connection(self):
        """Create and return database connection"""
        try:
            return pymysql.connect(
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", 3306)),
                database=os.getenv("DB_NAME", ""),
                user=os.getenv("DB_USER", ""),
                password=os.getenv("DB_PASSWORD", ""),
                charset='utf8mb4',
                cursorclass=DictCursor,
                autocommit=True
            )
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise


# Create global instance
db = DatabaseConnection()