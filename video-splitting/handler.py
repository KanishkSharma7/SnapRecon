import os
import boto3
import json
import subprocess
import logging
from urllib.parse import unquote_plus

# Initialize the S3 client and logger
s3Client = boto3.client('s3')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    # Get the bucket name and object key from the event
    inputBucket = os.environ['INPUT_BUCKET']
    outputBucket = os.environ['OUTPUT_BUCKET']
    logger.info("Event: %s", event)
    
    for record in event['Records']:
        # Extract the uploaded video file name from the event
        videoKey = unquote_plus(record['s3']['object']['key'])
        videoBasename = os.path.basename(videoKey)  # Extract filename only
        videoName, _ = os.path.splitext(videoBasename)  # Remove extension
        
        # Generate a two-digit number based on video name (for example purposes)
        videoNumber = int(videoName.split('_')[1])  # Assuming videoName has format test_XX
        formattedVideoName = f"test_{videoNumber:02d}"  # Ensure two-digit format
        
        logger.info("Processing file: %s from bucket: %s", videoKey, inputBucket)
        downloadDir = f'/tmp/{videoBasename}'
        outputImagePath = f'/tmp/{formattedVideoName}.jpg'  # Path for the output image
        
        try:
            # Download the video file from the input S3 bucket
            s3Client.download_file(inputBucket, videoKey, downloadDir)
            logger.info("Downloaded video to %s", downloadDir)
            
            # Extract a single frame as an image
            extractSingleFrame(downloadDir, outputImagePath)
            
            # Upload the extracted frame to the output S3 bucket with correct naming convention
            uploadImageToS3(outputImagePath, outputBucket, f"{formattedVideoName}.jpg")

            # Invoke the face-recognition Lambda function asynchronously
            lambda_client = boto3.client('lambda')
            try:
                print("Invoking next lambda function")
                face_recognition_payload = {
                    "bucket": outputBucket,        # Bucket where the frame was uploaded
                    "key": f"{formattedVideoName}.jpg"  # Key of the uploaded frame
                }
                lambda_client.invoke(
                    FunctionName='face-recognition',
                    InvocationType='Event',  # Asynchronous invocation
                    Payload=json.dumps(face_recognition_payload)
                )
                print("Invoked face-recognition function asynchronously")
            except Exception as e:
                print(f"Error invoking face-recognition function: {e}")
                return {
                    'statusCode': 500,
                    'body': json.dumps('Error invoking face-recognition function.')
                }

            return {
                'statusCode': 200,
                'body': json.dumps(f'Successfully processed video {formattedVideoName} and uploaded frame to {outputBucket}.')
            }
            
        except Exception as e:
            logger.error("Error processing video %s: %s", videoKey, str(e))
    
    return {
        'statusCode': 200,
        'body': 'Video processing complete'
    }

def extractSingleFrame(input_path, output_image_path):
    ffmpegPath = "/opt/bin/ffmpeg"
    # Run ffmpeg command to extract a single frame
    command = [
        ffmpegPath, "-i", input_path,  # Input video file
        "-vf", "select=eq(n\\,0)",  # Select the first frame
        "-vframes", "1",            # Limit to one frame
        output_image_path,          # Output image path
        "-y"                        # Overwrite if exists
    ]
    try:
        logger.info("Running ffmpeg command: %s", " ".join(command))
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("ffmpeg output: %s", result.stdout.decode("utf-8"))
        logger.info("Single frame extracted successfully")
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg failed with error: %s", e.stderr.decode("utf-8"))
        raise

def uploadImageToS3(image_path, bucket_name, s3_key):
    try:
        # Upload the image to the specified S3 bucket with correct naming convention (e.g., test_XX.jpg)
        s3Client.upload_file(image_path, bucket_name, s3_key)
        logger.info("Uploaded image: %s to bucket: %s", s3_key, bucket_name)
    except Exception as e:
        logger.error("Failed to upload %s to S3: %s", s3_key, e)
        raise