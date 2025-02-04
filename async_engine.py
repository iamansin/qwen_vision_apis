import asyncio
import time
import torch
import uuid
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from fastapi import HTTPException
import logging
from typing_extensions import List, Union
from faster_whisper import WhisperModel
import concurrent.futures
import numpy as np
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncEngine_Image:
    """Handles request batching and optimized batch inferencing."""

    def __init__(self, model_name: str , system_prompt: str, max_new_token: int, batch_size: int, max_wait_time: float):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.system_prompt = system_prompt
        self.max_new_token = max_new_token
        self.request_queue = asyncio.Queue()
        self.shutdown_event = asyncio.Event()
        self.background_task = None
        self.model, self.processor = self.load_model(model_name)
        self._start_background_loops()

    def load_model(self, model_name):
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
        

    def _start_background_loops(self):
        """Start the background inference loops."""
        self.background_task = asyncio.create_task(self._batch_inference_loop())

    async def _batch_inference_loop(self):
        """Continuously process incoming image requests in batches."""
        while not self.shutdown_event.is_set():
            batch = []
            start_time = time.time()

            while len(batch) < self.batch_size:
                try:
                    request = await asyncio.wait_for(self.request_queue.get(), timeout=0.09)
                    batch.append(request)
                except asyncio.TimeoutError:
                    if batch and (time.time() - start_time) >= self.max_wait_time:
                        break
                    continue

            if not batch:
                continue

            try:
                images = [img for req in batch for img in req[1]]
                response_events = [req[2] for req in batch]

                responses = await asyncio.to_thread(self._run_inference, images)
                idx = 0
                for i, (request_id, req_images, response_event) in enumerate(batch):
                    num_images = len(req_images)
                    result = {"request_id": request_id, "response": responses[idx:idx + num_images]}
                    response_events[i].set_result(result)
                    idx += num_images
            except Exception as e:
                for response_event in response_events:
                    response_event.set_exception(e)
                    

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
        """Queue an image request and return results once processed."""
        request_id = str(uuid.uuid4())
        response_event = asyncio.get_running_loop().create_future()
        await self.request_queue.put((request_id, images, response_event))
        return await response_event


    async def shutdown(self):
        logger.info("Shutting down AsyncEngine...")
        self.shutdown_event.set()
        logger.info("Freeing Model Space")
        del self.model
        del self.processor
        del self.whisper_client
        torch.cuda.empty_cache()
        logger.info("AsyncEngine shut down successfully.")
        
        


class AsyncEngine_Audio:
    """Handles request batching and optimized batch inferencing for audio transcription."""

    def __init__(self, max_new_token: int, batch_size: int, max_wait_time: float, whisper_model_path: str):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_new_token = max_new_token
        self.audio_request_queue = asyncio.Queue()
        self.shutdown_event = asyncio.Event()
        
        # Create a thread pool for CPU-bound operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # Initialize Whisper model with optimal settings
        self.whisper_client = self._initialize_whisper_model(whisper_model_path)
        
        # Add this line to start background loops
        self._start_background_loops()

    def _initialize_whisper_model(self, model_path: str):
        try:
            logging.info("Loading optimized Whisper model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Optimize model settings
            model = WhisperModel(
                model_path,
                device=device,
                compute_type="float16",
                cpu_threads=4,            # Optimize CPU thread usage
                num_workers=2,            # Parallel workers for data loading
            )
            
            # Warm up the model (optional, but can help reduce first-time latency)
            logging.info("Whisper model loaded successfully")
            return model
            
        except Exception as e:
            logging.error(f"Error initializing Whisper model: {str(e)}")
            raise e

    def _start_background_loops(self):
        """Start the background inference loops."""
        # Create a background task for audio batch processing
        self.audio_background_task = asyncio.create_task(self._audio_batch_inference_loop())

    async def _audio_batch_inference_loop(self):
        """Continuously process incoming audio requests in batches."""
        while not self.shutdown_event.is_set():
            batch = []
            start_time = time.time()

            # Collect batch with timeout
            while len(batch) < self.batch_size:
                try:
                    request = await asyncio.wait_for(
                        self.audio_request_queue.get(), 
                        timeout=self.max_wait_time
                    )
                    batch.append(request)
                except asyncio.TimeoutError:
                    if batch:  # Process partial batch
                        break
                    continue

            if not batch:
                continue

            try:
                audio_files = [req[1] for req in batch]
                response_events = [req[2] for req in batch]

                # Process audio files concurrently
                responses = await self._process_audio_batch(audio_files)
                
                for i, (request_id, _, response_event) in enumerate(batch):
                    result = {
                        "request_id": request_id,
                        "response": responses[i]
                    }
                    response_event.set_result(result)
                    
            except Exception as e:
                logging.error(f"Batch processing error: {str(e)}")
                for response_event in response_events:
                    response_event.set_exception(e)

    async def _process_audio_batch(self, audio_files: List[str]) -> List[str]:
        """Process a batch of audio files concurrently."""
        try:
            # Create tasks for each audio file
            tasks = [self._transcribe_single_file(audio_file) for audio_file in audio_files]
            
            # Wait for all transcriptions to complete
            return await asyncio.gather(*tasks)
            
        except Exception as e:
            logging.error(f"Batch processing error: {str(e)}")
            raise HTTPException(status_code=500, detail="Error transcribing audio batch")

    async def _transcribe_single_file(self, audio_file: str) -> str:
        """Transcribe a single audio file."""
        try:
            # Run transcription in thread pool
            segments, info = await asyncio.to_thread(
                self.whisper_client.transcribe,
                audio_file,
                language="en",
                task="translate",
                beam_size=1
            )
            
            return " ".join([segment.text for segment in segments]).strip()
        except Exception as e:
            logging.error(f"Error transcribing file {audio_file}: {str(e)}")
            raise

    async def enqueue_audio_request(self, audio_file: str):
        """Queue an audio request and return results once processed."""
        request_id = str(uuid.uuid4())
        response_event = asyncio.get_running_loop().create_future()
        await self.audio_request_queue.put((request_id, audio_file, response_event))
        return await response_event

    async def shutdown(self):
        """Gracefully shut down the engine."""
        logging.info("Shutting down AsyncEngine_Audio...")
        self.shutdown_event.set()
        
        # Wait for background task to complete
        if hasattr(self, 'audio_background_task'):
            await self.audio_background_task
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Clean up resources
        del self.whisper_client
        torch.cuda.empty_cache()
        logging.info("AsyncEngine_Audio shut down successfully.")




