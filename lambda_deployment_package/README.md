# Lambda Function Deployment Package

This directory contains the Lambda function for processing S3 events.

## Deployment Steps

1. **Create deployment package:**
   ```bash
   cd lambda_deployment_package
   pip install -r requirements.txt -t .
   zip -r lambda_function.zip lambda_function.py boto3* botocore*
   ```

2. **Create Lambda function via AWS CLI:**
   ```bash
   aws lambda create-function \
     --function-name deepfake-detection-s3-trigger \
     --runtime python3.10 \
     --role arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-execution-role \
     --handler lambda_function.lambda_handler \
     --zip-file fileb://lambda_function.zip \
     --timeout 60 \
     --memory-size 256 \
     --environment Variables="{SNS_TOPIC_ARN=arn:aws:sns:us-east-1:ACCOUNT:topic-name,SQS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/ACCOUNT/queue-name,BUCKET_NAME=deepfakedetection}"
   ```

3. **Configure S3 bucket trigger:**
   ```bash
   aws s3api put-bucket-notification-configuration \
     --bucket deepfakedetection \
     --notification-configuration file://s3-notification-config.json
   ```

4. **Or via AWS Console:**
   - Go to S3 → deepfakedetection bucket → Properties → Event notifications
   - Add notification: All object create events → Lambda function → deepfake-detection-s3-trigger

## Required IAM Permissions

The Lambda execution role needs:
- `s3:GetObject`
- `s3:HeadObject`
- `sns:Publish`
- `sqs:SendMessage`
- `logs:CreateLogGroup`
- `logs:CreateLogStream`
- `logs:PutLogEvents`

