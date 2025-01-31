import logging
import time
from typing import List, Union
from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
from PIL import Image
import asyncio
from async_engine import AsyncEngine
from utils import process_image_file, ImageAnalysisRequest, ImageAnalysisResponse, SYSTEM_PROMPT
# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
MAX_IMAGE_SIZE = (1024, 1024)
BATCH_SIZE = 5
MAX_WAIT_TIME = 0.5

    
engine = AsyncEngine(model_name = MODEL_NAME,
                     system_prompt=SYSTEM_PROMPT, 
                     max_new_token=500,
                     batch_size=BATCH_SIZE, 
                     max_wait_time=MAX_WAIT_TIME)

app = FastAPI()

@app.post("/analyze/", response_model=ImageAnalysisResponse)
async def analyze_images(images: Union[List[UploadFile],UploadFile] = File(...)):
    """Single endpoint for image analysis. Returns request ID and results once completed."""
    
    start_processing_time = time.time()
    if not isinstance(images,list):
        images = [images]
    logger.info(f"Received {len(images)} images for analysis")
    try:
        pil_images = await asyncio.gather(*(process_image_file(image,MAX_IMAGE_SIZE) for image in images))

        # Enqueue the request and wait for the response
        response = await engine.enqueue_request(pil_images)
        response = ImageAnalysisResponse(response=response.get("response"),
                                         request_id=response.get("request_id"), 
                                         request_processing_time= time.time()-start_processing_time,)
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing images")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
