# Deployment Guide - Deepfake Detection System

This guide covers deploying the Deepfake Detection System with Docker, Jenkins CI/CD, S3 storage, and Lambda functions.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Docker Deployment](#docker-deployment)
3. [Jenkins CI/CD Setup](#jenkins-cicd-setup)
4. [AWS S3 Configuration](#aws-s3-configuration)
5. [Lambda Function Setup](#lambda-function-setup)
6. [Environment Variables](#environment-variables)

## Prerequisites

- Docker and Docker Compose installed
- AWS Account with appropriate permissions
- Jenkins server (optional, for CI/CD)
- AWS CLI configured

## Docker Deployment

### Build Docker Image

```bash
docker build -t deepfake-detection:latest .
```

### Run with Docker Compose

```bash
# Set environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
export S3_BUCKET=deepfakedetection

# Start services
docker-compose up -d
```

### Run Standalone Container

```bash
docker run -d \
  -p 5000:5000 \
  -e S3_ENABLED=true \
  -e S3_BUCKET=deepfakedetection \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e AWS_DEFAULT_REGION=us-east-1 \
  --name deepfake-app \
  deepfake-detection:latest
```

## Jenkins CI/CD Setup

### 1. Install Required Jenkins Plugins

- Docker Pipeline
- AWS Steps
- Credentials Binding

### 2. Configure AWS Credentials in Jenkins

1. Go to Jenkins → Manage Jenkins → Credentials
2. Add AWS credentials (Access Key ID and Secret Access Key)
3. Note the credentials ID

### 3. Configure Jenkinsfile

Update the Jenkinsfile with:
- Your ECR repository URL
- ECS cluster and service names
- AWS region
- S3 bucket name

### 4. Create Jenkins Pipeline

1. New Item → Pipeline
2. Configure → Pipeline → Definition: Pipeline script from SCM
3. Repository URL: Your Git repository
4. Script Path: Jenkinsfile

### 5. Build Pipeline

The pipeline will:
- Build Docker image
- Run tests
- Push to ECR
- Deploy to ECS/Fargate
- Upload artifacts to S3

## AWS S3 Configuration

### 1. Create S3 Bucket

```bash
aws s3 mb s3://deepfakedetection --region us-east-1
```

### 2. Configure Bucket Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowLambdaAccess",
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::deepfakedetection/*"
    }
  ]
}
```

### 3. Enable Versioning (Optional)

```bash
aws s3api put-bucket-versioning \
  --bucket deepfakedetection \
  --versioning-configuration Status=Enabled
```

### 4. Configure Lifecycle Rules

```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket deepfakedetection \
  --lifecycle-configuration file://lifecycle-config.json
```

## Lambda Function Setup

### 1. Create IAM Role for Lambda

```bash
aws iam create-role \
  --role-name lambda-s3-execution-role \
  --assume-role-policy-document file://trust-policy.json

aws iam attach-role-policy \
  --role-name lambda-s3-execution-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
  --role-name lambda-s3-execution-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
```

### 2. Create SNS Topic (Optional)

```bash
aws sns create-topic --name deepfake-detection-notifications
```

### 3. Create SQS Queue (Optional)

```bash
aws sqs create-queue --queue-name deepfake-video-processing
```

### 4. Package Lambda Function

```bash
cd lambda_deployment_package
pip install -r requirements.txt -t .
zip -r lambda_function.zip lambda_function.py boto3* botocore*
```

### 5. Deploy Lambda Function

```bash
aws lambda create-function \
  --function-name deepfake-detection-s3-trigger \
  --runtime python3.10 \
  --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-s3-execution-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda_function.zip \
  --timeout 60 \
  --memory-size 256 \
  --environment Variables="{SNS_TOPIC_ARN=arn:aws:sns:us-east-1:ACCOUNT:deepfake-detection-notifications,SQS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/ACCOUNT/deepfake-video-processing,BUCKET_NAME=deepfakedetection}"
```

### 6. Configure S3 Event Notification

Via AWS Console:
1. Go to S3 → deepfakedetection bucket
2. Properties → Event notifications
3. Create event notification:
   - Event types: All object create events
   - Destination: Lambda function → deepfake-detection-s3-trigger

Or via CLI:
```bash
aws s3api put-bucket-notification-configuration \
  --bucket deepfakedetection \
  --notification-configuration file://s3-notification-config.json
```

## Environment Variables

### Application Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `S3_ENABLED` | Enable S3 storage | `true` |
| `S3_BUCKET` | S3 bucket name | `deepfakedetection` |
| `AWS_ACCESS_KEY_ID` | AWS access key | Required |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | Required |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |
| `FLASK_ENV` | Flask environment | `production` |

### Lambda Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SNS_TOPIC_ARN` | SNS topic for notifications | Optional |
| `SQS_QUEUE_URL` | SQS queue for processing | Optional |
| `BUCKET_NAME` | S3 bucket name | Yes |

## Testing

### Test Docker Container

```bash
docker run --rm -p 5000:5000 deepfake-detection:latest
curl http://localhost:5000/api/status
```

### Test S3 Integration

```bash
python -c "from s3_storage import S3Storage; s3 = S3Storage(); print('S3 connection OK')"
```

### Test Lambda Function

```bash
aws lambda invoke \
  --function-name deepfake-detection-s3-trigger \
  --payload file://test-event.json \
  response.json
```

## Monitoring

### CloudWatch Logs

- Application logs: `/aws/ecs/deepfake-detection`
- Lambda logs: `/aws/lambda/deepfake-detection-s3-trigger`

### S3 Metrics

Monitor S3 bucket metrics in CloudWatch:
- Number of objects
- Bucket size
- Request metrics

## Troubleshooting

### Docker Issues

- Check logs: `docker logs deepfake-detection-app`
- Verify environment variables: `docker exec deepfake-detection-app env`

### S3 Issues

- Verify credentials: `aws sts get-caller-identity`
- Check bucket permissions: `aws s3 ls s3://deepfakedetection`

### Lambda Issues

- Check CloudWatch logs
- Verify IAM permissions
- Test with sample event

## Security Best Practices

1. Use IAM roles instead of access keys when possible
2. Enable S3 bucket encryption
3. Use VPC endpoints for S3 access
4. Enable CloudTrail for audit logging
5. Rotate credentials regularly
6. Use least privilege IAM policies

## Cost Optimization

1. Enable S3 lifecycle policies for old files
2. Use S3 Intelligent-Tiering
3. Set up CloudWatch alarms for costs
4. Use Lambda reserved concurrency to control costs

