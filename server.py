import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import io
from contextlib import asynccontextmanager
from typing import Tuple

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
MAX_IMAGE_SIZE = (1024, 1024)  # Maximum image dimensions
MAX_BATCH_SIZE = 1  # Adjust based on your GPU memory
DEFAULT_SYSTEM_PROMPT = """
Analyze the provided image and the patient's basic description of their condition. Then, describe the patient's visible condition based on the image, focusing on the following:

1. Identifying the specific area(s) of concern
2. Describing the visible symptoms in detail

Important Notes:
- This is not professional medical advice
- Avoid assumptions about diagnosis
- Maintain empathetic and respectful language
- Do not mention diagnosis
"""

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_model_settings(model):
    """Apply various optimization settings to the model."""
    if torch.cuda.is_available():
        # Enable memory efficient attention if available
        if hasattr(model.config, 'use_memory_efficient_attention') and \
           model.config.use_memory_efficient_attention:
            model.config.use_memory_efficient_attention = True
            logger.info("Enabled memory efficient attention")

        # Enable Flash Attention if available
        if hasattr(model.config, 'use_flash_attention') and \
           model.config.use_flash_attention:
            model.config.use_flash_attention = True
            logger.info("Enabled Flash Attention")

    # Enable model optimization flags
    model.eval()  # Ensure model is in evaluation mode
    return model

def resize_image(image: Image.Image, max_size: Tuple[int, int]) -> Image.Image:
    """Resize image while maintaining aspect ratio if it exceeds max dimensions."""
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        logger.info(f"Resized image to {image.size}")
    return image

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Loading model and processor...")
        
        # Load model with optimizations
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            device_map="auto",  # Enable automatic device mapping
            cache_dir="/content"
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
    # Clean up resources
    torch.cuda.empty_cache()
    logger.info("Resources cleaned up.")

async def process_image_file(file: UploadFile) -> Image.Image:
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        # Resize image if necessary
        image = resize_image(image, MAX_IMAGE_SIZE)
        logger.info("Image processed successfully.")
        return image
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file")

@torch.inference_mode()  # More efficient than with torch.no_grad()
def generate_response(image: Image.Image, system_prompt: str, model, processor) -> str:
    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": system_prompt},
            ]
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        # Process inputs with optimized batch size
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            max_length=2048,  # Add maximum sequence length
        ).to(DEVICE)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,  # Disable sampling for faster inference
            num_beams=1,      # Use greedy decoding
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        logger.info("Response generated successfully.")
        return response

    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating response")

@app.post("/analyze/")
async def analyze_image(
    image: UploadFile = File(...),
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
):
    try:
        pil_image = await process_image_file(image)
        result = await run_in_threadpool(
            generate_response,
            pil_image,
            system_prompt,
            app.state.model,
            app.state.processor
        )
        logger.info("Image analysis completed successfully.")
        return {"analysis": result}
    except HTTPException as he:
        logger.error(f"HTTP error: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
