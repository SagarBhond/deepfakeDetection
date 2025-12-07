# Quick Lambda Deployment Guide - Copy & Paste Method

## ✅ Yes! You can copy-paste directly into Lambda Console

The `lambda_function.py` uses only built-in libraries and `boto3` (which is already available in Lambda runtime), so you can paste it directly!

## 🚀 Step-by-Step: Deploy via Lambda Console

### Step 1: Open AWS Lambda Console

1. Go to [AWS Lambda Console](https://console.aws.amazon.com/lambda/)
2. Click **"Create function"**
3. Choose **"Author from scratch"**
4. Function name: `deepfake-detection-s3-trigger`
5. Runtime: **Python 3.10** or **Python 3.11**
6. Architecture: **x86_64**
7. Click **"Create function"**

### Step 2: Paste Your Code

1. In the Lambda function page, scroll down to **"Code source"**
2. Delete the default code
3. Open `lambda_function.py` from your project
4. **Copy ALL the code** (Ctrl+A, Ctrl+C)
5. **Paste it** into the Lambda code editor (Ctrl+V)
6. Click **"Deploy"** button (top right)

### Step 3: Configure Environment Variables

1. Go to **"Configuration"** tab
2. Click **"Environment variables"** in left sidebar
3. Click **"Edit"** → **"Add environment variable"**
4. Add these variables:

| Key | Value | Description |
|-----|-------|-------------|
| `BUCKET_NAME` | `deepfakeddetection` | Your S3 bucket name |
| `SNS_TOPIC_ARN` | `arn:aws:sns:ap-south-1:ACCOUNT:topic-name` | (Optional) SNS topic for notifications |
| `SQS_QUEUE_URL` | `https://sqs.ap-south-1.amazonaws.com/ACCOUNT/queue-name` | (Optional) SQS queue URL |

**Note:** SNS_TOPIC_ARN and SQS_QUEUE_URL are optional. You can leave them empty if you don't have them yet.

5. Click **"Save"**

### Step 4: Configure IAM Role (Permissions)

1. Go to **"Configuration"** tab
2. Click **"Permissions"** in left sidebar
3. Click on the **Execution role** name
4. This opens IAM in a new tab
5. Click **"Add permissions"** → **"Attach policies"**
6. Attach these policies:
   - `AmazonS3ReadOnlyAccess` (or create custom policy with S3 access)
   - `AWSLambdaBasicExecutionRole` (for CloudWatch logs)
   - If using SNS: `AmazonSNSFullAccess`
   - If using SQS: `AmazonSQSFullAccess`

### Step 5: Configure Basic Settings

1. Go to **"Configuration"** tab
2. Click **"General configuration"** → **"Edit"**
3. Set:
   - **Timeout**: `60` seconds (or more if processing large files)
   - **Memory**: `256` MB (or more if needed)
4. Click **"Save"**

### Step 6: Add S3 Trigger

1. Go to **"Configuration"** tab
2. Click **"Triggers"** in left sidebar
3. Click **"Add trigger"**
4. Select **"S3"**
5. Configure:
   - **Bucket**: `deepfakeddetection`
   - **Event type**: `All object create events` (or specific events)
   - **Prefix**: `videos/` (optional - to trigger only on videos folder)
   - **Suffix**: (leave empty or specify like `.mp4`)
6. Check **"Recursive invocation"** warning (if shown)
7. Click **"Add"**

### Step 7: Test Your Function

1. Go back to **"Code"** tab
2. Click **"Test"** button
3. Create a new test event:
   - Event name: `S3Test`
   - Event JSON: Use the sample below
4. Click **"Save"** then **"Test"**

**Sample Test Event:**
```json
{
  "Records": [
    {
      "eventVersion": "2.1",
      "eventSource": "aws:s3",
      "awsRegion": "ap-south-1",
      "eventTime": "2024-01-01T00:00:00.000Z",
      "eventName": "ObjectCreated:Put",
      "s3": {
        "bucket": {
          "name": "deepfakeddetection"
        },
        "object": {
          "key": "videos/20240101/test-video.mp4",
          "size": 1024000
        }
      }
    }
  ]
}
```

## 📋 Complete Code to Copy-Paste

The entire `lambda_function.py` file is ready to paste. Here's what it includes:

- ✅ S3 event processing
- ✅ Video upload handling
- ✅ Result file processing
- ✅ Model upload handling
- ✅ SNS notifications (optional)
- ✅ SQS messaging (optional)
- ✅ Error handling and logging

## 🔍 Verify It's Working

1. **Upload a test file to S3:**
   ```bash
   aws s3 cp test.mp4 s3://deepfakeddetection/videos/test.mp4
   ```

2. **Check Lambda logs:**
   - Go to Lambda function → **"Monitor"** tab
   - Click **"View CloudWatch logs"**
   - You should see logs of the function execution

3. **Check S3 bucket:**
   - Go to S3 console
   - Verify file was uploaded
   - Check if Lambda was triggered

## ⚠️ Important Notes

1. **Region**: Make sure Lambda function is in **ap-south-1** (same as your S3 bucket)
2. **Permissions**: Lambda needs permission to read from S3 bucket
3. **Timeout**: Set appropriate timeout for large file processing
4. **Memory**: Increase if processing large files

## 🆘 Troubleshooting

### Function not triggering?
- Check S3 bucket region matches Lambda region
- Verify trigger is configured correctly
- Check Lambda execution role has S3 permissions

### Getting errors?
- Check CloudWatch logs for detailed error messages
- Verify environment variables are set correctly
- Ensure IAM role has all required permissions

### Code not working?
- Make sure you copied the ENTIRE file
- Check Python runtime version (3.10 or 3.11)
- Verify indentation is correct

## 📦 Alternative: Deployment Package Method

If you prefer using a deployment package (for version control or CI/CD):

1. See `lambda_deployment_package/README.md` for instructions
2. Use `zip` file upload in Lambda console
3. Or use AWS CLI to deploy

---

**That's it! Your Lambda function is ready to process S3 events.** 🎉

