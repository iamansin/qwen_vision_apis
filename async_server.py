import logging
import time
import io
from typing import List, Tuple, Union
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
from contextlib import asynccontextmanager
import asyncio
# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
MAX_IMAGE_SIZE = (1024, 1024)
SYSTEM_PROMPT = """
Analyze the provided image and describe the visible condition in detail.
- Identify specific areas of concern.
- Describe symptoms without making a diagnosis.
- Maintain empathetic and respectful language.
"""

logger.info(f"Using device: {DEVICE}")

#class ImageAnalysisRequest(BaseModel):
 #   system_prompt: str = DEFAULT_SYSTEM_PROMPT

class ImageAnalysisResponse(BaseModel):
    analyses: List[str]

def optimize_model_settings(model):
    logger.info("Applying optimizations to the model")
    model.to(DEVICE)
    model.eval()
    return model

def resize_image(image: Image.Image, max_size: Tuple[int, int]) -> Image.Image:
    logger.info("Resizing the image")
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        logger.info(f"Resized image to {image.size}")
    return image

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Loading model and processor...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=DTYPE, device_map="auto"
        )
        model = optimize_model_settings(model)
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        app.state.model = model
        app.state.processor = processor
        logger.info("Model and processor loaded successfully.")
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise
    yield
    torch.cuda.empty_cache()
    logger.info("Resources cleaned up.")

app = FastAPI(lifespan=lifespan)

async def process_image_file(file: UploadFile) -> Image.Image:
    try:
        logger.info("Processing the image")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return resize_image(image, MAX_IMAGE_SIZE)
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file")

@torch.inference_mode()
def generate_responses(images: List[Image.Image], system_prompt: str, model, processor) -> List[str]:
    try:
        logger.info("Creating batched messages for the images")
        messages_batch = [
            [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": system_prompt}]}]
            for img in images
        ]
        logger.info("Applying chat_template for response generation")
        text_batch = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch]
        processed_info = [process_vision_info(msg) for msg in messages_batch]
        image_inputs = [info[0] for info in processed_info]
        video_inputs = [info[1] if info[1] is not None else [] for info in processed_info]
        print(video_inputs)
        if not any(image_inputs):
            raise HTTPException(status_code=400, detail="No valid images found for processing.")
        logger.info("Processng images by the autoprocessor")
        inputs = processor(
            text=text_batch,
            images=list(image_inputs),
            padding="longest",
            return_tensors="pt",
        ).to(DEVICE)
        logger.info("Generating response for the image/images")
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            num_beams=1,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating responses")

@app.post("/analyze_batch/")
async def analyze_images(images:Union[List[UploadFile], UploadFile] = File(...),):
    try:
        if not isinstance(images,list):
            logger.info("Single image detected")
            images = [images]
        start_time = time.time()
        logger.info("Pssing images for batched image processing")
        pil_images = await asyncio.gather(*(process_image_file(img) for img in images))
        logger.info("Running model for inferencing")
        results = await run_in_threadpool(generate_responses, pil_images, SYSTEM_PROMPT, app.state.model, app.state.processor)
        logger.info(f"Total processing time: {time.time() - start_time:.2f} sec")
        return ImageAnalysisResponse(analyses=results)
    except HTTPException as he:
        logger.error(f"HTTP error: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=5000)
