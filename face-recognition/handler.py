import os
import json
import boto3
import torch
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

os.environ['TORCH_HOME'] = '/tmp'
s3Client = boto3.client('s3')

def downloadFromS3(bucket, key, local_path):
    try:
        s3Client.download_file(bucket, key, local_path)
    except Exception as e:
        print(f"Error downloading from S3: {str(e)}")
        raise e

def uploadToS3(local_path, bucket, key):
    try:
        s3Client.upload_file(local_path, bucket, key)
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        raise e

def faceRecognition(image_path):
    # Initialize MTCNN and ResNet models
    mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    
    # Read and process image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Detect face
    face, prob = mtcnn(img_pil, return_prob=True)
    
    if face is not None:
        # Load the saved embeddings
        saved_data = torch.load('/tmp/data.pt')
        embedding_list = saved_data[0]
        name_list = saved_data[1]
        
        # Get embedding for the detected face
        emb = resnet(face.unsqueeze(0)).detach()
        
        # Compare with saved embeddings
        dist_list = [torch.dist(emb, emb_db).item() for emb_db in embedding_list]
        idx_min = dist_list.index(min(dist_list))
        
        return name_list[idx_min]
    return None

def handler(event, context):
    try:
        eventBody = json.loads(event['body']) if isinstance(event.get('body'), str) else event
        bucketName = eventBody['bucket']
        imageName = eventBody['key']
        
        asuId = "1226213666"
        outputBucket = f"{asuId}-output"
        dataBucket = f"{asuId}-data"
        
        print("Downloading data.pt from data bucket")
        downloadFromS3(dataBucket, 'data.pt', '/tmp/data.pt')
        
        print("Downloading input image")
        localImageDir = f"/tmp/{imageName}"
        downloadFromS3(bucketName, imageName, localImageDir)
        
        print("Processing image")
        recName = faceRecognition(localImageDir)
        
        testNum = os.path.splitext(imageName)[0].split('_')[1]  
        
        print("Creating output text file")
        outputFileName = f"test_{testNum}.txt"
        outputPath = f"/tmp/{outputFileName}"
        
        with open(outputPath, 'w') as f:
            f.write(recName if recName else "No face detected")
        
        print("Uploading result to output bucket")
        uploadToS3(outputPath, outputBucket, outputFileName)
        
        print(f"Successfully processed {imageName} and saved result to {outputBucket}/{outputFileName}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Face recognition completed successfully',
                'recName': recName,
                'output_file': outputFileName
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
        	})
	}