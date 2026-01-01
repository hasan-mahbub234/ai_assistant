"""
S3-compatible storage implementation (for production)
"""
import boto3
from botocore.exceptions import ClientError
import logging
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class S3Storage:
    """S3-compatible object storage"""
    
    def __init__(
        self,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        region: str = "auto",
    ):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Ensure bucket exists, create if not"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                # Bucket doesn't exist, create it
                try:
                    if self.s3_client.meta.region_name == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={
                                'LocationConstraint': self.s3_client.meta.region_name
                            }
                        )
                    logger.info(f"Created bucket: {self.bucket_name}")
                except ClientError as create_error:
                    logger.error(f"Failed to create bucket: {str(create_error)}")
                    raise
            else:
                logger.error(f"Failed to check bucket: {str(e)}")
                raise
    
    async def save_file(
        self,
        content: bytes,
        filename: str,
        user_id: int,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Save file to S3
        """
        try:
            s3_key = f"{user_id}/{filename}"
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=content,
                ContentType=content_type or 'application/octet-stream',
                Metadata={
                    'user_id': str(user_id),
                    'original_filename': filename,
                }
            )
            
            logger.info(f"File saved to S3: {s3_key}")
            return s3_key
            
        except ClientError as e:
            logger.error(f"Failed to save file to S3: {str(e)}")
            raise
    
    async def read_file(self, s3_key: str) -> bytes:
        """
        Read file from S3
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key,
            )
            
            return response['Body'].read()
            
        except ClientError as e:
            logger.error(f"Failed to read file from S3: {str(e)}")
            raise
    
    async def delete_file(self, s3_key: str) -> bool:
        """
        Delete file from S3
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key,
            )
            logger.info(f"File deleted from S3: {s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete file from S3: {str(e)}")
            return False
    
    async def file_exists(self, s3_key: str) -> bool:
        """
        Check if file exists in S3
        """
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key,
            )
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise
    
    async def get_presigned_url(
        self,
        s3_key: str,
        expires_in: int = 3600,
    ) -> str:
        """
        Generate presigned URL for temporary access
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key,
                },
                ExpiresIn=expires_in,
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {str(e)}")
            raise
    
    async def list_user_files(self, user_id: int) -> List[str]:
        """
        List all files for a user
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{user_id}/",
            )
            
            if 'Contents' not in response:
                return []
            
            return [obj['Key'] for obj in response['Contents']]
            
        except ClientError as e:
            logger.error(f"Failed to list user files: {str(e)}")
            return []