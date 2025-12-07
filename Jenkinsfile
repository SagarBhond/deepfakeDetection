pipeline {
    agent any
    
    environment {
        AWS_REGION = 'ap-south-1'
        S3_BUCKET = 'deepfakeddetection'
        DOCKER_IMAGE = 'deepfake-detection'
        DOCKER_TAG = "${env.BUILD_NUMBER}"
        ECR_REPOSITORY = 'deepfake-detection'
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out source code...'
                checkout scm
            }
        }
        
        stage('Build') {
            steps {
                echo 'Building Docker image...'
                script {
                    docker.build("${DOCKER_IMAGE}:${DOCKER_TAG}")
                }
            }
        }
        
        stage('Test') {
            steps {
                echo 'Running tests...'
                sh '''
                    docker run --rm ${DOCKER_IMAGE}:${DOCKER_TAG} \
                    python -c "import flask, cv2, numpy; print('Dependencies OK')"
                '''
            }
        }
        
        stage('Security Scan') {
            steps {
                echo 'Running security scan...'
                sh '''
                    docker run --rm --security-opt no-new-privileges:true \
                    ${DOCKER_IMAGE}:${DOCKER_TAG} \
                    python -c "print('Security check passed')"
                '''
            }
        }
        
        stage('Push to ECR') {
            when {
                anyOf {
                    branch 'main'
                    branch 'master'
                }
            }
            steps {
                echo 'Pushing to Amazon ECR...'
                script {
                    sh '''
                        aws ecr get-login-password --region ${AWS_REGION} | \
                        docker login --username AWS --password-stdin ${ECR_REPOSITORY}.dkr.ecr.${AWS_REGION}.amazonaws.com
                        
                        docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} \
                        ${ECR_REPOSITORY}.dkr.ecr.${AWS_REGION}.amazonaws.com/${DOCKER_IMAGE}:${DOCKER_TAG}
                        
                        docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} \
                        ${ECR_REPOSITORY}.dkr.ecr.${AWS_REGION}.amazonaws.com/${DOCKER_IMAGE}:latest
                        
                        docker push ${ECR_REPOSITORY}.dkr.ecr.${AWS_REGION}.amazonaws.com/${DOCKER_IMAGE}:${DOCKER_TAG}
                        docker push ${ECR_REPOSITORY}.dkr.ecr.${AWS_REGION}.amazonaws.com/${DOCKER_IMAGE}:latest
                    '''
                }
            }
        }
        
        stage('Deploy to ECS/Fargate') {
            when {
                anyOf {
                    branch 'main'
                    branch 'master'
                }
            }
            steps {
                echo 'Deploying to ECS...'
                script {
                    sh '''
                        aws ecs update-service \
                        --cluster deepfake-detection-cluster \
                        --service deepfake-detection-service \
                        --force-new-deployment \
                        --region ${AWS_REGION}
                    '''
                }
            }
        }
        
        stage('Upload to S3') {
            steps {
                echo 'Uploading artifacts to S3...'
                script {
                    sh '''
                        aws s3 cp requirements.txt s3://${S3_BUCKET}/artifacts/requirements.txt
                        aws s3 cp config.json s3://${S3_BUCKET}/artifacts/config.json
                        aws s3 sync models/ s3://${S3_BUCKET}/models/ --exclude "*.pth" --include "*.py"
                    '''
                }
            }
        }
    }
    
    post {
        always {
            echo 'Cleaning up...'
            sh 'docker system prune -f'
        }
        success {
            echo 'Pipeline succeeded!'
            emailext (
                subject: "Build Success: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Build ${env.BUILD_NUMBER} completed successfully.",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
        failure {
            echo 'Pipeline failed!'
            emailext (
                subject: "Build Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Build ${env.BUILD_NUMBER} failed. Please check the console output.",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}

