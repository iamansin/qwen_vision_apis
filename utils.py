from PIL import Image
from textwrap import dedent
from fastapi import  UploadFile, HTTPException
from pydantic import BaseModel
import io
from typing_extensions import List, Tuple
import logging 
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




