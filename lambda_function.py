"""
AWS Lambda Function for Deepfake Detection S3 Event Processing
Triggers when files are uploaded to S3 bucket: deepfakedetection
"""

import json
import boto3
import os
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
sns_client = boto3.client('sns')
sqs_client = boto3.client('sqs')

# Environment variables
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN', '')
SQS_QUEUE_URL = os.environ.get('SQS_QUEUE_URL', '')
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'deepfakeddetection')

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for S3 events
    
    Args:
        event: S3 event data
        context: Lambda context
        
    Returns:
        Response dictionary
    """
    logger.info(f"Received event: {json.dumps(event)}")
    
    try:
        # Process each S3 record in the event
        for record in event.get('Records', []):
            # Check if it's an S3 event
            if record.get('eventSource') != 'aws:s3':
                continue
            
            # Extract S3 event details
            s3_event = record.get('s3', {})
            bucket_name = s3_event.get('bucket', {}).get('name')
            object_key = s3_event.get('object', {}).get('key')
            event_name = record.get('eventName', '')
            
            logger.info(f"Processing S3 event: {event_name} for {bucket_name}/{object_key}")
            
            # Process based on event type
            if 'ObjectCreated' in event_name:
                handle_object_created(bucket_name, object_key)
            elif 'ObjectRemoved' in event_name:
                handle_object_removed(bucket_name, object_key)
            else:
                logger.info(f"Unhandled event type: {event_name}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'S3 events processed successfully',
                'processed_records': len(event.get('Records', []))
            })
        }
    
    except Exception as e:
        logger.error(f"Error processing S3 event: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'Failed to process S3 event'
            })
        }

def handle_object_created(bucket_name: str, object_key: str):
    """
    Handle S3 object creation event
    
    Args:
        bucket_name: S3 bucket name
        object_key: S3 object key
    """
    logger.info(f"Handling object creation: {bucket_name}/{object_key}")
    
    # Get object metadata
    try:
        response = s3_client.head_object(Bucket=bucket_name, Key=object_key)
        file_size = response['ContentLength']
        content_type = response.get('ContentType', '')
        last_modified = response['LastModified']
        
        # Determine file type and process accordingly
        if object_key.startswith('videos/'):
            process_video_upload(bucket_name, object_key, file_size, content_type)
        elif object_key.startswith('results/'):
            process_result_upload(bucket_name, object_key, file_size)
        elif object_key.startswith('models/'):
            process_model_upload(bucket_name, object_key, file_size)
        else:
            logger.info(f"Unhandled object type: {object_key}")
    
    except Exception as e:
        logger.error(f"Error handling object creation: {str(e)}")

def process_video_upload(bucket_name: str, object_key: str, file_size: int, content_type: str):
    """
    Process video file upload
    
    Args:
        bucket_name: S3 bucket name
        object_key: S3 object key
        file_size: File size in bytes
        content_type: Content type
    """
    logger.info(f"Processing video upload: {object_key}")
    
    # Create notification message
    message = {
        'event_type': 'video_uploaded',
        'bucket': bucket_name,
        'key': object_key,
        'size': file_size,
        'content_type': content_type,
        'timestamp': datetime.utcnow().isoformat(),
        'action': 'trigger_analysis'
    }
    
    # Send to SQS queue for processing
    if SQS_QUEUE_URL:
        try:
            sqs_client.send_message(
                QueueUrl=SQS_QUEUE_URL,
                MessageBody=json.dumps(message)
            )
            logger.info(f"Sent video processing message to SQS: {object_key}")
        except Exception as e:
            logger.error(f"Error sending to SQS: {str(e)}")
    
    # Send notification via SNS
    if SNS_TOPIC_ARN:
        try:
            sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Subject=f'New Video Uploaded: {object_key}',
                Message=json.dumps(message, indent=2)
            )
            logger.info(f"Sent notification via SNS: {object_key}")
        except Exception as e:
            logger.error(f"Error sending SNS notification: {str(e)}")

def process_result_upload(bucket_name: str, object_key: str, file_size: int):
    """
    Process result file upload
    
    Args:
        bucket_name: S3 bucket name
        object_key: S3 object key
        file_size: File size in bytes
    """
    logger.info(f"Processing result upload: {object_key}")
    
    # Download and parse result file
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        result_data = json.loads(response['Body'].read())
        
        # Extract key information
        prediction = result_data.get('prediction', 'Unknown')
        confidence = result_data.get('confidence', 0.0)
        is_fake = result_data.get('is_fake', False)
        
        # Create notification
        message = {
            'event_type': 'result_uploaded',
            'bucket': bucket_name,
            'key': object_key,
            'prediction': prediction,
            'confidence': confidence,
            'is_fake': is_fake,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Send notification
        if SNS_TOPIC_ARN:
            sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Subject=f'Analysis Result: {prediction} (Confidence: {confidence:.2%})',
                Message=json.dumps(message, indent=2)
            )
            logger.info(f"Sent result notification: {object_key}")
    
    except Exception as e:
        logger.error(f"Error processing result upload: {str(e)}")

def process_model_upload(bucket_name: str, object_key: str, file_size: int):
    """
    Process model file upload
    
    Args:
        bucket_name: S3 bucket name
        object_key: S3 object key
        file_size: File size in bytes
    """
    logger.info(f"Processing model upload: {object_key}")
    
    message = {
        'event_type': 'model_uploaded',
        'bucket': bucket_name,
        'key': object_key,
        'size': file_size,
        'timestamp': datetime.utcnow().isoformat(),
        'action': 'update_model_cache'
    }
    
    if SNS_TOPIC_ARN:
        try:
            sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Subject=f'New Model Uploaded: {object_key}',
                Message=json.dumps(message, indent=2)
            )
            logger.info(f"Sent model upload notification: {object_key}")
        except Exception as e:
            logger.error(f"Error sending model notification: {str(e)}")

def handle_object_removed(bucket_name: str, object_key: str):
    """
    Handle S3 object removal event
    
    Args:
        bucket_name: S3 bucket name
        object_key: S3 object key
    """
    logger.info(f"Handling object removal: {bucket_name}/{object_key}")
    
    message = {
        'event_type': 'object_removed',
        'bucket': bucket_name,
        'key': object_key,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    if SNS_TOPIC_ARN:
        try:
            sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Subject=f'Object Removed: {object_key}',
                Message=json.dumps(message, indent=2)
            )
        except Exception as e:
            logger.error(f"Error sending removal notification: {str(e)}")

