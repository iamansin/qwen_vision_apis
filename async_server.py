import logging
import time
from typing import List, Union
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
import torch
from PIL import Image
import asyncio
from async_engine import AsyncEngine_Audio, AsyncEngine_Image
from utils import process_image_file, ImageAnalysisRequest, ImageAnalysisResponse, SYSTEM_PROMPT, process_audio_file
from contextlib import asynccontextmanager
from utils import download_s3_folder
import concurrent.futures
import numpy as np

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
WHISPER_MODEL_PATH =  download_s3_folder(bucket_name="speech-to-text-local-model", s3_folder="Speech_To_large_Model",
                               local_folder="Speech_To_Text_Large_Model")
MAX_IMAGE_SIZE = (1024, 1024)
BATCH_SIZE = 5
MAX_WAIT_TIME = 0.5

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    app.state.image_engine = AsyncEngine_Image(model_name=MODEL_NAME,
                         system_prompt=SYSTEM_PROMPT,
                         max_new_token=500,
                         batch_size=BATCH_SIZE,
                         max_wait_time=MAX_WAIT_TIME,
                         )
    
    app.state.audio_engine = AsyncEngine_Audio(
                         max_new_token=500,
                         batch_size=BATCH_SIZE,
                         max_wait_time=MAX_WAIT_TIME,
                         whisper_model_path=WHISPER_MODEL_PATH
                         )
    
    yield
    
    await asyncio.gather[app.state.audio_engine.shutdown(), app.state.image_engine.shutdown()]

app = FastAPI(lifespan=lifespan)

@app.post("/itt_endpoint", response_model=ImageAnalysisResponse)
async def analyze_images(request: Request , images: Union[List[UploadFile], UploadFile] = File(...)):
    """Single endpoint for image analysis. Returns request ID and results once completed."""
    start_processing_time = time.time()
    if not isinstance(images, list):
        images = [images]
    logger.info(f"Received {len(images)} images for analysis")
    try:
        pil_images = await asyncio.gather(*(process_image_file(image, MAX_IMAGE_SIZE) for image in images))
        response = await request.app.state.image_engine.enqueue_request(pil_images)
        response = ImageAnalysisResponse(response=response.get("response"),
                                         request_id=response.get("request_id"),
                                         request_processing_time=time.time() - start_processing_time)
        return response
    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing images")

@app.post("/stt_endpoint")
async def voice_prescription(request: Request, file: UploadFile = File(...)):
    """Optimized audio processing endpoint."""
    try:
        # Add performance monitoring
        start_time = time.time()
        
        # Process audio file
        filepath = await process_audio_file(file)
        
        # Get transcription
        response = await request.app.state.audio_engine.enqueue_audio_request(filepath)
        
        # Clean up temporary file
        try:
            os.remove(filepath)
        except Exception as e:
            logger.warning(f"Error removing temporary file {filepath}: {e}")
        
        # Log performance metrics
        processing_time = time.time() - start_time
        logger.info(f"Request processed in {processing_time:.2f} seconds")
        
        return {
            "transcription": response.get("response"),
            "request_id": response.get("request_id"),
            "processing_time": processing_time
        }
        
    except HTTPException as he:
        logger.error(f"HTTP error: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=7000)
