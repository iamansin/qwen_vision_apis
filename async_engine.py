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
import os
import tempfile
from utils import process_audio_stream
import torchaudio
import asyncio
import logging
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple

import torch
import torchaudio
from fastapi import HTTPException
from faster_whisper import WhisperModel
from io import BytesIO


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncEngine_Image:
    """Handles request batching and optimized batch inferencing."""

    def __init__(self, model_name: str , system_prompt: str, max_new_token: int, batch_size: int, max_wait_time: float, gpu_utilization : float =0.6):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.system_prompt = system_prompt
        self.max_new_token = max_new_token
        self.gpu_utilization = gpu_utilization
        self.request_queue = asyncio.Queue()
        self.shutdown_event = asyncio.Event()
        self.background_task = None
        self.model, self.processor = self.load_model(model_name)
        self._start_background_loops()

    def load_model(self, model_name):
        try:    
                _ , total_mem = torch.cuda.mem_get_info()
                total_mem_GB = total_mem / (1024 ** 3)  

                max_memory_GB = int(total_mem_GB * self.gpu_utilization)  
                max_memory = {0: f"{max_memory_GB}GB"}
                logger.info(f"Allocating {max_memory_GB}GB to Qwen Vision Model... ")
                logger.info("Loading model and processor...")
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name, torch_dtype=torch.float16, device_map="auto",
                    max_memory = max_memory,
                ).eval()
                processor = AutoProcessor.from_pretrained(model_name)
                logger.info("Model and processor loaded successfully.")
                available_mem, _ = torch.cuda.mem_get_info()
                logger.info(f"Memory available after loading the model {int(available_mem / (1024 ** 3))}")
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
        torch.cuda.empty_cache()
        logger.info("AsyncEngine shut down successfully.")


class AsyncEngine_Audio:
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac'}

    def __init__(self, max_new_token: int, batch_size: int, max_wait_time: float, whisper_model_path: str):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_new_token = max_new_token
        self.audio_request_queue = asyncio.Queue()
        self.shutdown_event = asyncio.Event()
        self.audio_background_task = None
        self.whisper_client = self.load_whisper_model(whisper_model_path)
        self._start_background_loops()

    def load_whisper_model(self, model_path):
        try:
            logger.info("Loading Whisper model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            whisper_client = WhisperModel(
                model_path,
                device=device,
                compute_type="float16",
                download_root=None,
                local_files_only=False,
                cpu_threads=8,
                num_workers=2,
            )
            logger.info("Whisper model loaded successfully.")
            return whisper_client
        except Exception as e:
            logger.error(f"Error initializing Whisper model: {str(e)}")
            raise e

    def _start_background_loops(self):
        self.audio_background_task = asyncio.create_task(self._audio_batch_inference_loop())

    async def _audio_batch_inference_loop(self):
        while not self.shutdown_event.is_set():
            batch = []
            start_time = time.time()

            while len(batch) < self.batch_size:
                try:
                    request = await asyncio.wait_for(self.audio_request_queue.get(), timeout=0.09)
                    batch.append(request)
                except asyncio.TimeoutError:
                    if batch and (time.time() - start_time) >= self.max_wait_time:
                        break
                    continue

            if not batch:
                continue

            try:
                tasks = []
                for request_id, audio_data, file_extension, response_event in batch:
                    task = asyncio.create_task(self._process_single_audio(
                        request_id, audio_data, file_extension, response_event
                    ))
                    tasks.append(task)

                await asyncio.gather(*tasks)

            except Exception as e:
                logger.error(f"Batch processing error: {str(e)}")
                for _, _, _, response_event in batch:
                    if not response_event.done():
                        response_event.set_exception(e)

    async def _process_single_audio(self, request_id: str, audio_data: bytes, file_extension: str, response_event):
            try:
                response = await asyncio.to_thread(
                    self._run_single_audio_inference, audio_data, file_extension
                )
                result = {"request_id": request_id, "response": response}
                response_event.set_result(result)
            except Exception as e:
                logger.error(f"Error processing audio {request_id}: {str(e)}")
                response_event.set_exception(e)

    def _run_single_audio_inference(self, audio_data: bytes, file_extension: str) -> str:
        try:
            waveform, sample_rate = process_audio_stream(audio_data)
            # Create a temporary file with the correct extension
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=True) as temp_file:
                torchaudio.save(temp_file.name, waveform, sample_rate)
                segments, _ = self.whisper_client.transcribe(temp_file.name, language="en", task="translate",beam_size=1,vad_filter=True)


            return " ".join([segment.text for segment in segments]).strip()

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise HTTPException(status_code=500, detail="Error transcribing audio")

    async def enqueue_audio_request(self, audio_data: bytes, file_extension: str):
        if file_extension.lower() not in self.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format. Supported formats are: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        request_id = str(uuid.uuid4())
        response_event = asyncio.get_running_loop().create_future()
        await self.audio_request_queue.put((request_id, audio_data, file_extension, response_event))
        return await response_event

    async def shutdown(self):
        logger.info("Shutting down AsyncEngine...")
        self.shutdown_event.set()
        if self.audio_background_task:
            await self.audio_background_task
        logger.info("Freeing Model Space")
        del self.whisper_client
        torch.cuda.empty_cache()
        logger.info("AsyncEngine shut down successfully.")
