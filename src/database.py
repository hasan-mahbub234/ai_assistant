import pymysql
from src.config import config
import logging

logger = logging.getLogger(__name__)

def get_db_connection():
    """Create database connection for Vercel AI service"""
    try:
        connection = pymysql.connect(
            host=config.DB_HOST,
            database=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            port=config.DB_PORT,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=10,
            read_timeout=30,
            write_timeout=30
        )
        
        logger.info("✅ Successfully connected to MySQL database from Vercel")
        return connection
        
    except Exception as e:
        logger.error(f"❌ Error connecting to MySQL database: {e}")
        raise

def close_db_connection(connection):
    """Close database connection"""
    if connection and connection.open:
        connection.close()
        logger.info("🔌 MySQL connection closed")