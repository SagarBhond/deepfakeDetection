# Quick Start Guide - Docker, Jenkins, S3 & Lambda

## 🚀 Quick Deployment

### 1. Build and Run with Docker

```bash
# Build image
docker build -t deepfake-detection:latest .

# Run with docker-compose
docker-compose up -d

# Or run standalone
docker run -d -p 5000:5000 \
  -e S3_ENABLED=true \
  -e S3_BUCKET=deepfakeddetection \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  deepfake-detection:latest
```

### 2. Setup S3 Bucket

```bash
# Create bucket
aws s3 mb s3://deepfakeddetection --region ap-south-1

# Verify
aws s3 ls s3://deepfakeddetection
```

### 3. Deploy Lambda Function

```bash
cd lambda_deployment_package

# Install dependencies and package
pip install -r requirements.txt -t .
zip -r lambda_function.zip lambda_function.py boto3* botocore*

# Deploy
aws lambda create-function \
  --function-name deepfake-detection-s3-trigger \
  --runtime python3.10 \
  --role arn:aws:iam::ACCOUNT:role/lambda-execution-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda_function.zip \
  --timeout 60 \
  --memory-size 256
```

### 4. Configure S3 Trigger

```bash
# Via AWS Console:
# S3 → deepfakeddetection → Properties → Event notifications
# Add: All object create events → Lambda → deepfake-detection-s3-trigger
```

### 5. Setup Jenkins Pipeline

1. Install Jenkins plugins: Docker Pipeline, AWS Steps
2. Configure AWS credentials in Jenkins
3. Create new Pipeline job
4. Point to your Git repository with Jenkinsfile
5. Build!

## 📁 Project Structure

```
deepfakeDetection/
├── Dockerfile                    # Docker image definition
├── docker-compose.yml           # Docker Compose configuration
├── Jenkinsfile                  # CI/CD pipeline
├── basic_web_app.py            # Main Flask application (S3 integrated)
├── s3_storage.py               # S3 storage handler
├── lambda_function.py          # Lambda function for S3 events
├── lambda_deployment_package/  # Lambda deployment package
│   ├── lambda_function.py
│   ├── requirements.txt
│   └── README.md
├── requirements.txt            # Python dependencies (includes boto3)
├── DEPLOYMENT_GUIDE.md        # Detailed deployment guide
└── .dockerignore              # Docker ignore file
```

## 🔧 Environment Variables

Required for S3 integration:
- `S3_ENABLED=true`
- `S3_BUCKET=deepfakeddetection`
- `AWS_ACCESS_KEY_ID=your_key`
- `AWS_SECRET_ACCESS_KEY=your_secret`
- `AWS_DEFAULT_REGION=ap-south-1`

## ✅ Verify Setup

1. **Test Docker**: `curl http://localhost:5000/api/status`
2. **Test S3**: Upload a video via web interface, check S3 bucket
3. **Test Lambda**: Upload file to S3, check CloudWatch logs
4. **Test Jenkins**: Push to Git, verify pipeline runs

## 📚 Documentation

- Full deployment guide: `DEPLOYMENT_GUIDE.md`
- Lambda setup: `lambda_deployment_package/README.md`
- Main README: `README.md`

