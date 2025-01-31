import asyncio
import time
import torch
import uuid
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from fastapi import HTTPException
import logging 
from typing_extensions import List
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsyncEngine:
    """Handles request batching and optimized batch inferencing."""

    def __init__(self, model_name: str, system_prompt: str, max_new_token:int , batch_size: int, max_wait_time: float):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.system_prompt = system_prompt
        self.max_new_token = max_new_token
        self.request_queue = asyncio.Queue()
        self.model , self.processor = self.load_model(model_name)
        self._start_background_loop()
        
       
    def load_model(self,model_name):
        try:
            logger.info("Loading model and processor...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            ).eval()  
            processor = AutoProcessor.from_pretrained(model_name)
            logger.info("Model and processor loaded successfully.")
            return model, processor
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            torch.cuda.empty_cache() 
            raise e
        
    def _start_background_loop(self):
        """Start the background inference loop."""
        asyncio.create_task(self._batch_inference_loop())

    async def _batch_inference_loop(self):
        """Continuously process incoming requests in batches."""
        while True:
            batch = []
            start_time = time.time()

            while len(batch) < self.batch_size:
                try:
                    # Try fetching from the queue with a timeout
                    request = await asyncio.wait_for(self.request_queue.get(), timeout=0.09)
                    batch.append(request)
                except asyncio.TimeoutError:
                    # If the queue is empty and we have pending requests, process them after max_wait_time
                    if batch and (time.time() - start_time) >= self.max_wait_time:
                        break
                    continue  # Continue waiting for new requests

            if not batch:
                continue 
            
            if batch:
                images = [img for req in batch for img in req[0]]
                response_events = [req[1] for req in batch]

                try:
                    responses = await asyncio.to_thread(self._run_inference, images)
                    
                except Exception as e:
                    for response_event in response_events:
                        response_event.set_exception(e)
    
                # Distribute responses back to respective requests
                idx = 0
                for i, (request_id, req_images, response_event) in enumerate(batch):
                    num_images = len(req_images)
                    result = {"request_id": request_id, "response": responses[idx:idx + num_images]}
                    response_events[i].set_result(result)  # Send response directly
                    idx += num_images

    @torch.inference_mode()
    def _run_inference(self, images: List[Image.Image]) -> List[str]:
        try:
            start_generation_time = time.time()
            logger.info("Processing batch of %d images", len(images))

            messages_batch = [
                [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": self.system_prompt}]}]
                for img in images
            ]
            
            logger.info("Applying chat_template for response generation")
            processed_info = list(map(process_vision_info, messages_batch))
            image_inputs = [info[0] for info in processed_info if info[0] is not None]

            if not image_inputs:
                raise HTTPException(status_code=400, detail="No valid images found.")

            inputs = self.processor(
                text=[self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch],
                images=image_inputs,
                padding="longest",
                return_tensors="pt",
            ).to("cuda")

            logger.info("Generating response for %d images", len(image_inputs))
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_token, do_sample=False, num_beams=1,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            logger.info("Batch processed in %.2f seconds", time.time() - start_generation_time)
            return self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise HTTPException(status_code=500, detail="Error generating responses")

    async def enqueue_request(self, images: List[Image.Image]):
        """Queue a request and return results once processed."""
        request_id = str(uuid.uuid4())  # Generate unique request ID
        response_event = asyncio.get_running_loop().create_future()
        await self.request_queue.put((request_id, images, response_event))
        return await response_event  # Wait for batch inference completion and return result
