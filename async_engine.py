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


# class AsyncEngine_Audio:
#     """Handles request batching and optimized batch inferencing for audio."""
#
#     def __init__(self, max_new_token: int, batch_size: int, max_wait_time: float, whisper_model_path: str):
#         self.batch_size = batch_size
#         self.max_wait_time = max_wait_time
#         self.max_new_token = max_new_token
#         self.audio_request_queue = asyncio.Queue()
#         self.shutdown_event = asyncio.Event()
#         self.audio_background_task = None
#         self.processing_semaphore = asyncio.Semaphore(10)  # Limit concurrent processing
#         self.whisper_client = self.load_whisper_model(whisper_model_path)
#         self._start_background_loops()
#
#     def load_whisper_model(self, model_path):
#         try:
#             logger.info("Loading Whisper model...")
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#             whisper_client = WhisperModel(
#                 model_path,
#                 device=device,
#                 compute_type="float16",
#                 download_root=None,
#                 local_files_only=False,
#                 cpu_threads=0
#                 # Maximum number of batches to queue
#             )
#
#
#             logger.info("Whisper model loaded successfully.")
#             return whisper_client
#         except Exception as e:
#             logger.error(f"Error initializing Whisper model: {str(e)}")
#             raise e
#
#     def _start_background_loops(self):
#         """Start the background inference loops."""
#         self.audio_background_task = asyncio.create_task(self._audio_batch_inference_loop())
#
#     async def _audio_batch_inference_loop(self):
#         """Continuously process incoming audio requests in batches."""
#         while not self.shutdown_event.is_set():
#             batch = []
#             start_time = time.time()
#
#             # Collect batch
#             while len(batch) < self.batch_size:
#                 try:
#                     request = await asyncio.wait_for(self.audio_request_queue.get(), timeout=0.09)
#                     batch.append(request)
#                 except asyncio.TimeoutError:
#                     if batch and (time.time() - start_time) >= self.max_wait_time:
#                         break
#                     continue
#
#             if not batch:
#                 continue
#
#             # Process batch in parallel
#             try:
#                 tasks = []
#                 for request_id, audio_file, response_event in batch:
#                     task = asyncio.create_task(self._process_single_audio(
#                         request_id, audio_file, response_event
#                     ))
#                     tasks.append(task)
#
#                 await asyncio.gather(*tasks)
#
#             except Exception as e:
#                 logger.error(f"Batch processing error: {str(e)}")
#                 for _, _, response_event in batch:
#                     if not response_event.done():
#                         response_event.set_exception(e)
#
#     async def _process_single_audio(self, request_id: str, audio_file: str, response_event):
#         """Process a single audio file with semaphore limiting."""
#         async with self.processing_semaphore:
#             try:
#                 response = await asyncio.to_thread(
#                     self._run_single_audio_inference, audio_file
#                 )
#                 result = {"request_id": request_id, "response": response}
#                 response_event.set_result(result)
#             except Exception as e:
#                 logger.error(f"Error processing audio {request_id}: {str(e)}")
#                 response_event.set_exception(e)
#             finally:
#                 # Cleanup temporary audio file
#                 try:
#                     if os.path.exists(audio_file):
#                         os.remove(audio_file)
#                 except Exception as e:
#                     logger.error(f"Error removing temporary file {audio_file}: {str(e)}")
#
#     def _run_single_audio_inference(self, audio_file: str) -> str:
#         """Run inference on a single audio file."""
#         try:
#             segments, info = self.whisper_client.transcribe(
#                 audio_file,
#                 language="en",
#                 task="translate",
#                 beam_size=1,
#
#             )
#             return " ".join([segment.text for segment in segments]).strip()
#         except Exception as e:
#             logger.error(f"Transcription error: {str(e)}")
#             raise HTTPException(status_code=500, detail="Error transcribing audio")
#
#     async def enqueue_audio_request(self, audio_file: str):
#         """Queue an audio request and return results once processed."""
#         request_id = str(uuid.uuid4())
#         response_event = asyncio.get_running_loop().create_future()
#         await self.audio_request_queue.put((request_id, audio_file, response_event))
#         return await response_event
#
#     async def shutdown(self):
#         """Shutdown the engine and clean up resources."""
#         logger.info("Shutting down AsyncEngine...")
#         self.shutdown_event.set()
#         if self.audio_background_task:
#             await self.audio_background_task
#         logger.info("Freeing Model Space")
#         del self.whisper_client
#         torch.cuda.empty_cache()
#         logger.info("AsyncEngine shut down successfully.")
#
class AsyncEngine_Audio:
    """
    Optimized async engine for processing audio transcription requests with better GPU utilization
    and concurrent request handling.
    """
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac'}

    def __init__(self, max_new_token: int, batch_size: int, max_wait_time: float, whisper_model_path: str):
        """
        Initialize the AsyncEngine_Audio with optimized parameters.

        Args:
            max_new_token (int): Maximum number of tokens to generate
            batch_size (int): Size of batches for processing
            max_wait_time (float): Maximum time to wait for batch completion
            whisper_model_path (str): Path to the Whisper model
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_new_token = max_new_token
        self.audio_request_queue = asyncio.Queue()
        self.shutdown_event = asyncio.Event()
        self.audio_background_task = None

        # Configure thread pool for CPU-bound operations
        self.processing_pool = ThreadPoolExecutor(max_workers=4)

        # Batch processing controls
        self.current_batch = []
        self.batch_lock = asyncio.Lock()

        # Load the model with optimized settings
        self.whisper_client = self.load_whisper_model(whisper_model_path)

        # Start background processing
        self._start_background_loops()

    def load_whisper_model(self, model_path: str) -> WhisperModel:
        """
        Load and configure the Whisper model with optimized settings.

        Args:
            model_path (str): Path to the Whisper model

        Returns:
            WhisperModel: Configured Whisper model instance
        """
        try:
            logger.info("Loading Whisper model with optimized settings...")
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Get GPU memory info
            if device == "cuda":
                free_mem, total_mem = torch.cuda.mem_get_info()
                gpu_mem_gb = free_mem / (1024 ** 3)
                logger.info(f"Available GPU memory: {gpu_mem_gb:.2f} GB")

                # Adjust batch size based on available memory
                optimal_gpu_batch_size = max(1, min(8, int(gpu_mem_gb / 1.5)))
                logger.info(f"Setting GPU batch size to: {optimal_gpu_batch_size}")
            else:
                optimal_gpu_batch_size = 1

            whisper_client = WhisperModel(
                 model_path,
                 device=device,
                 compute_type="float16",
                 download_root=None,
                 local_files_only=False,
                 cpu_threads=0
                 # Maximum number of batches to queue
             )

            logger.info("Whisper model loaded successfully with optimized settings")
            return whisper_client

        except Exception as e:
            logger.error(f"Error initializing Whisper model: {str(e)}")
            raise e

    def _start_background_loops(self):
        """Start the background processing loops."""
        self.audio_background_task = asyncio.create_task(self._audio_batch_inference_loop())

    async def _audio_batch_inference_loop(self):
        """
        Main loop for processing audio batches with optimized GPU utilization.
        """
        while not self.shutdown_event.is_set():
            batch = []
            start_time = time.time()

            # Collect batch with timeout
            async with self.batch_lock:
                while len(batch) < self.batch_size and (time.time() - start_time) < self.max_wait_time:
                    try:
                        request = await asyncio.wait_for(
                            self.audio_request_queue.get(),
                            timeout=0.1
                        )
                        batch.append(request)
                    except asyncio.TimeoutError:
                        break

            if not batch:
                await asyncio.sleep(0.1)
                continue

            # Process batch in parallel using GPU batching
            try:
                # Organize batch data
                audio_data = []
                file_extensions = []
                response_events = []
                request_ids = []

                for request_id, data, ext, event in batch:
                    audio_data.append(data)
                    file_extensions.append(ext)
                    response_events.append(event)
                    request_ids.append(request_id)

                # Parallel audio preprocessing
                preprocessed_audio = await asyncio.gather(
                    *[self._preprocess_audio(data, ext)
                      for data, ext in zip(audio_data, file_extensions)]
                )

                # Run GPU batch inference
                results = await self._run_batch_inference(preprocessed_audio)

                # Set results for all requests in batch
                for request_id, response_event, result in zip(request_ids, response_events, results):
                    if not response_event.done():
                        response_event.set_result({
                            "request_id": request_id,
                            "response": result
                        })

            except Exception as e:
                logger.error(f"Batch processing error: {str(e)}")
                for event in response_events:
                    if not event.done():
                        event.set_exception(e)

    async def _preprocess_audio(self, audio_data: bytes, file_extension: str) -> Tuple[torch.Tensor, int, str]:
        """
        Preprocess audio data asynchronously using thread pool.

        Args:
            audio_data (bytes): Raw audio data
            file_extension (str): Audio file extension

        Returns:
            Tuple[torch.Tensor, int, str]: Processed waveform, sample rate, and file extension
        """
        return await asyncio.get_event_loop().run_in_executor(
            self.processing_pool,
            self._preprocess_audio_sync,
            audio_data,
            file_extension
        )

    def _preprocess_audio_sync(self, audio_data: bytes, file_extension: str) -> Tuple[torch.Tensor, int, str]:
        """
        Synchronous audio preprocessing function.

        Args:
            audio_data (bytes): Raw audio data
            file_extension (str): Audio file extension

        Returns:
            Tuple[torch.Tensor, int, str]: Processed waveform, sample rate, and file extension
        """
        waveform, sample_rate = self._process_audio_stream(audio_data)
        return waveform, sample_rate, file_extension

    def _process_audio_stream(self, audio_bytes: bytes) -> Tuple[torch.Tensor, int]:
        """
        Process raw audio bytes into waveform and sample rate.

        Args:
            audio_bytes (bytes): Raw audio data

        Returns:
            Tuple[torch.Tensor, int]: Audio waveform and sample rate
        """
        try:
            waveform, sample_rate = torchaudio.load(BytesIO(audio_bytes))
            return waveform, sample_rate
        except Exception as e:
            logger.error(f"Error processing audio stream: {str(e)}")
            raise HTTPException(status_code=400, detail="Error processing audio data")

    async def _run_batch_inference(self, preprocessed_batch: List[Tuple[torch.Tensor, int, str]]) -> List[str]:
        """
        Run batch inference on GPU for multiple audio files.

        Args:
            preprocessed_batch (List[Tuple[torch.Tensor, int, str]]): Batch of preprocessed audio data

        Returns:
            List[str]: Batch transcription results
        """
        try:
            temp_files = []
            results = []

            # Create temporary files for batch processing
            for waveform, sample_rate, file_extension in preprocessed_batch:
                with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                    torchaudio.save(temp_file.name, waveform, sample_rate)
                    temp_files.append(temp_file.name)

            try:
                # Process batch on GPU
                segments_batch = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._process_batch_on_gpu,
                    temp_files
                )
                return segments_batch

            finally:
                # Clean up temporary files
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        logger.error(f"Error removing temporary file {temp_file}: {str(e)}")

        except Exception as e:
            logger.error(f"Batch inference error: {str(e)}")
            raise HTTPException(status_code=500, detail="Error in batch transcription")

    def _process_batch_on_gpu(self, file_paths: List[str]) -> List[str]:
        """
        Process multiple audio files on GPU in a single batch.

        Args:
            file_paths (List[str]): Paths to temporary audio files

        Returns:
            List[str]: Transcription results for each audio file
        """
        results = []

        for file_path in file_paths:
            try:
                segments, _ = self.whisper_client.transcribe(
                    file_path,
                    language="en",
                    task="translate",
                    #batch_size=self.batch_size,
                    vad_filter=True
                )
                results.append(" ".join([segment.text for segment in segments]).strip())
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                results.append("")

        return results

    async def enqueue_audio_request(self, audio_data: bytes, file_extension: str) -> Dict[str, Any]:
        """
        Enqueue an audio request for processing.

        Args:
            audio_data (bytes): Raw audio data
            file_extension (str): Audio file extension

        Returns:
            Dict[str, Any]: Response containing request ID and transcription
        """
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
        """Shutdown the engine and clean up resources."""
        logger.info("Shutting down AsyncEngine...")
        self.shutdown_event.set()

        if self.audio_background_task:
            await self.audio_background_task

        # Cleanup resources
        self.processing_pool.shutdown(wait=True)
        logger.info("Freeing Model Space")
        del self.whisper_client
        torch.cuda.empty_cache()
        logger.info("AsyncEngine shut down successfully.")
