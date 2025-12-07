"""
S3 Storage Integration for Deepfake Detection System
Handles all S3 operations for storing videos, results, and metadata
"""

import boto3
import json
import os
from datetime import datetime
from botocore.exceptions import ClientError
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class S3Storage:
    """S3 storage handler for deepfake detection system"""
    
    def __init__(self, bucket_name: str = 'deepfakeddetection', region: str = 'ap-south-1'):
        """
        Initialize S3 storage client
        
        Args:
            bucket_name: Name of the S3 bucket
            region: AWS region
        """
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.s3_resource = boto3.resource('s3', region_name=region)
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket {self.bucket_name} exists")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                try:
                    # Create bucket with region configuration
                    if self.region == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region}
                        )
                    logger.info(f"Created bucket {self.bucket_name}")
                except ClientError as create_error:
                    logger.error(f"Error creating bucket: {create_error}")
            else:
                logger.error(f"Error checking bucket: {e}")
    
    def upload_file(self, local_path: str, s3_key: str, content_type: Optional[str] = None) -> bool:
        """
        Upload a file to S3
        
        Args:
            local_path: Local file path
            s3_key: S3 object key (path in bucket)
            content_type: MIME type of the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            self.s3_client.upload_file(
                local_path,
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            logger.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Error uploading file: {e}")
            return False
    
    def upload_video(self, local_path: str, video_id: str) -> Optional[str]:
        """
        Upload video file to S3
        
        Args:
            local_path: Local video file path
            video_id: Unique identifier for the video
            
        Returns:
            S3 key if successful, None otherwise
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        s3_key = f"videos/{timestamp}/{video_id}"
        
        if self.upload_file(local_path, s3_key, content_type='video/mp4'):
            return s3_key
        return None
    
    def upload_result(self, result_data: Dict, result_id: str) -> Optional[str]:
        """
        Upload analysis result to S3
        
        Args:
            result_data: Result dictionary
            result_id: Unique identifier for the result
            
        Returns:
            S3 key if successful, None otherwise
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        s3_key = f"results/{timestamp}/{result_id}.json"
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json.dumps(result_data, indent=2),
                ContentType='application/json'
            )
            logger.info(f"Uploaded result to s3://{self.bucket_name}/{s3_key}")
            return s3_key
        except ClientError as e:
            logger.error(f"Error uploading result: {e}")
            return None
    
    def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Download a file from S3
        
        Args:
            s3_key: S3 object key
            local_path: Local file path to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
        except ClientError as e:
            logger.error(f"Error downloading file: {e}")
            return False
    
    def get_presigned_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate presigned URL for S3 object
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds
            
        Returns:
            Presigned URL or None
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None
    
    def list_objects(self, prefix: str = '') -> List[str]:
        """
        List objects in S3 bucket with given prefix
        
        Args:
            prefix: S3 key prefix
            
        Returns:
            List of object keys
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except ClientError as e:
            logger.error(f"Error listing objects: {e}")
            return []
    
    def delete_object(self, s3_key: str) -> bool:
        """
        Delete an object from S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Deleted s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Error deleting object: {e}")
            return False
    
    def get_object_metadata(self, s3_key: str) -> Optional[Dict]:
        """
        Get metadata for an S3 object
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Metadata dictionary or None
        """
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'].isoformat(),
                'content_type': response.get('ContentType', ''),
                'metadata': response.get('Metadata', {})
            }
        except ClientError as e:
            logger.error(f"Error getting metadata: {e}")
            return None
    
    def sync_results_to_s3(self, local_results_dir: str) -> int:
        """
        Sync local results directory to S3
        
        Args:
            local_results_dir: Local results directory path
            
        Returns:
            Number of files synced
        """
        synced_count = 0
        if not os.path.exists(local_results_dir):
            return synced_count
        
        for filename in os.listdir(local_results_dir):
            if filename.endswith('_result.json'):
                local_path = os.path.join(local_results_dir, filename)
                s3_key = f"results/{filename}"
                if self.upload_file(local_path, s3_key, content_type='application/json'):
                    synced_count += 1
        
        return synced_count

