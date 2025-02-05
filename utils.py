from PIL import Image
from datetime import time
from textwrap import dedent
from fastapi import UploadFile, HTTPException
from pydantic import BaseModel
import io
from typing_extensions import List, Tuple
import logging
import os
import uuid
import boto3
from dotenv import load_dotenv
import logging
import numpy as np
import torchaudio
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = dedent("""
Analyze the provided image and describe the visible condition in detail.
- Identify specific areas of concern.
- Describe symptoms without making a diagnosis.
- Maintain empathetic and respectful language.
""")

class ImageAnalysisRequest(BaseModel):
    system_prompt: str = SYSTEM_PROMPT

class ImageAnalysisResponse(BaseModel):
    analyses: List[str]
    
    
def resize_image(image: Image.Image, max_size: Tuple[int, int]) -> Image.Image:
    logger.info("Resizing the image")
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        logger.info(f"Resized image to {image.size}")
    return image

async def process_image_file(file: UploadFile, max_image_size) -> Image.Image:
    try:
        logger.info("Processing the image")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return resize_image(image, max_image_size)
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file")

# async def process_audio_file(file: UploadFile) -> str:
#     """Process audio file and save to temporary location."""
#     if not file.filename:
#         raise HTTPException(status_code=400, detail="No audio file detected")
#
#     temp_dir = os.path.join(os.getcwd(), "static", "temp_audio")
#     os.makedirs(temp_dir, exist_ok=True)
#
#     unique_filename = f"{uuid.uuid4()}_{file.filename}"
#     filepath = os.path.join(temp_dir, unique_filename)
#
#     try:
#         content = await file.read()
#         with open(filepath, "wb") as buffer:
#             buffer.write(content)
#         return filepath
#     except Exception as e:
#         logger.error(f"Error processing audio file: {str(e)}")
#         raise HTTPException(status_code=400, detail="Error processing audio file")
async def process_audio_data(file: UploadFile) -> tuple[bytes, str]:
    """Process audio file in memory and return data and extension."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No audio file detected")

    try:
        file_extension = os.path.splitext(file.filename)[1].lower()
        content = await file.read()
        return content, file_extension
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        raise HTTPException(status_code=400, detail="Error processing audio file")
def process_audio_stream(audio_bytes):
    """Convert raw audio bytes into a waveform and sample rate."""
    waveform, sample_rate = torchaudio.load(BytesIO(audio_bytes))
    return waveform, sample_rate

def load_aws_credentials():
    """
    Load AWS credentials from environment variables
    """
    load_dotenv()  # Load environment variables from .env file if it exists
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region_name = "ap-south-1"

    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError(
            "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")

    return aws_access_key_id, aws_secret_access_key, region_name

def download_s3_folder(bucket_name, s3_folder, local_folder):
    """
    Downloads a folder from an S3 bucket to your local machine,
    only if a local folder with the same name does not already exist.
    """
    aws_access_key_id, aws_secret_access_key, region_name = load_aws_credentials()

    s3 = boto3.client('s3',
                      aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key,
                      region_name=region_name)

    try:
        if os.path.exists(local_folder):
            logger.warning(f"Folder '{local_folder}' already exists locally. Skipping download.")
            return local_folder

        os.makedirs(local_folder, exist_ok=True)

        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)

        if 'Contents' in response:
            for obj in response['Contents']:
                s3_object_key = obj['Key']
                local_file_path = os.path.join(local_folder, os.path.relpath(s3_object_key, s3_folder))
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                s3.download_file(bucket_name, s3_object_key, local_file_path)
            logger.warning(f"Folder '{s3_folder}' downloaded from S3 to '{local_folder}' successfully!")
            return local_folder

        else:
            logger.error(f"Error: Folder '{s3_folder}' not found in bucket '{bucket_name}'")
            raise HTTPException(status_code=404, detail="Model folder not found in S3 bucket")

    except boto3.exceptions.Boto3Error as e:
        logger.error(f"Error accessing S3: {e}")
        raise HTTPException(status_code=500, detail="Error accessing S3")
    except OSError as e:
        logger.error(f"Error with local file system: {e}")
        raise HTTPException(status_code=500, detail="Error with local file system")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")




