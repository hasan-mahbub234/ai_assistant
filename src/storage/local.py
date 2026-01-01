"""
Local file storage implementation
"""
import os
import shutil
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from src.core.config import settings

logger = logging.getLogger(__name__)


class LocalStorage:
    """Local file storage"""
    
    def __init__(self):
        self.base_dir = Path(settings.UPLOAD_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_user_directory(self, user_id: int) -> Path:
        """Get user-specific directory"""
        user_dir = self.base_dir / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
    
    async def save_file(
        self,
        content: bytes,
        filename: str,
        user_id: int,
    ) -> str:
        """
        Save file to local storage
        """
        try:
            user_dir = self._get_user_directory(user_id)
            file_path = user_dir / filename
            
            # Write file
            with open(file_path, 'wb') as f:
                f.write(content)
            
            # Return relative path
            return str(file_path.relative_to(self.base_dir))
            
        except Exception as e:
            logger.error(f"Failed to save file: {str(e)}")
            raise
    
    async def read_file(self, file_path: str) -> bytes:
        """
        Read file from local storage
        """
        try:
            full_path = self.base_dir / file_path
            
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            with open(full_path, 'rb') as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"Failed to read file: {str(e)}")
            raise
    
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete file from local storage
        """
        try:
            full_path = self.base_dir / file_path
            
            if full_path.exists():
                os.remove(full_path)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete file: {str(e)}")
            return False
    
    async def file_exists(self, file_path: str) -> bool:
        """
        Check if file exists
        """
        full_path = self.base_dir / file_path
        return full_path.exists()
    
    async def get_file_size(self, file_path: str) -> int:
        """
        Get file size in bytes
        """
        full_path = self.base_dir / file_path
        
        if full_path.exists():
            return os.path.getsize(full_path)
        return 0
    
    async def list_user_files(self, user_id: int) -> List[str]:
        """
        List all files for a user
        """
        user_dir = self._get_user_directory(user_id)
        
        if not user_dir.exists():
            return []
        
        files = []
        for file_path in user_dir.iterdir():
            if file_path.is_file():
                files.append(str(file_path.relative_to(self.base_dir)))
        
        return files
    
    async def cleanup_old_files(self, days: int = 30) -> int:
        """
        Cleanup files older than specified days
        """
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        deleted_count = 0
        
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                file_path = Path(root) / file
                
                # Check if file is older than cutoff
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete old file {file_path}: {str(e)}")
        
        logger.info(f"Cleaned up {deleted_count} old files")
        return deleted_count