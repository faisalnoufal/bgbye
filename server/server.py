from fastapi import FastAPI, UploadFile, File, Response, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import io
import shutil
import time
import numpy as np
import tempfile
import uuid  
import os
import subprocess
import logging
import asyncio
from datetime import datetime, timedelta
import torch
from typing import Dict
from contextlib import contextmanager


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Create temp_videos folder if it doesn't exist
TEMP_VIDEOS_DIR = "temp_videos"
os.makedirs(TEMP_VIDEOS_DIR, exist_ok=True)

# Create a frames directory within temp_videos
FRAMES_DIR = os.path.join(TEMP_VIDEOS_DIR, "frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

# Add a dictionary to store processing status
processing_status = {}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def cleanup_old_videos():
    while True:
        current_time = datetime.now()
        for item in os.listdir(TEMP_VIDEOS_DIR):
            item_path = os.path.join(TEMP_VIDEOS_DIR, item)
            item_modified = datetime.fromtimestamp(os.path.getmtime(item_path))
            if current_time - item_modified > timedelta(minutes=10):
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    logger.info(f"Removed old file: {item_path}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    logger.info(f"Removed old directory: {item_path}")
        await asyncio.sleep(600)  # Run every 10 minutes

# Pre-load models - Tracer-B7, BASNet, and RMBG-2.0
# Import carvekit for tracer and basnet
carvekit_available = False
try:
    from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
    from carvekit.ml.wrap.basnet import BASNET
    from carvekit.ml.wrap.fba_matting import FBAMatting
    from carvekit.api.interface import Interface
    from carvekit.pipelines.postprocessing import MattingMethod
    from carvekit.pipelines.preprocessing import PreprocessingStub
    from carvekit.trimap.generator import TrimapGenerator
    
    def initialize_carvekit_model(seg_pipe_class, device='cpu'):
        model = Interface(
            pre_pipe=PreprocessingStub(),
            post_pipe=MattingMethod(
                matting_module=FBAMatting(device=device, input_tensor_size=2048, batch_size=1),
                trimap_generator=TrimapGenerator(),
                device=device
            ),
            seg_pipe=seg_pipe_class(device=device, batch_size=1)
        )
        return model
    
    carvekit_models = {
        'tracer': initialize_carvekit_model(TracerUniversalB7, device='cpu'),
        'basnet': initialize_carvekit_model(BASNET, device='cpu')
    }
    carvekit_available = True
    logger.info("Carvekit models loaded: Tracer-B7 and BASNet available")
except ImportError as e:
    logger.error(f"Carvekit not available: {e}. Cannot start server.")
    exit(1)

# Initialize RMBG-2.0 model
rmbg_available = False
try:
    from transformers import AutoModelForImageSegmentation
    from torchvision import transforms
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rmbg_model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
    rmbg_model.eval()
    rmbg_model.to(device)
    
    # RMBG-2.0 preprocessing transform
    rmbg_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    rmbg_available = True
    logger.info("RMBG-2.0 model loaded successfully")
except Exception as e:
    logger.warning(f"RMBG-2.0 not available: {e}")

available_models = "Tracer-B7, BASNet"
if rmbg_available:
    available_models += ", RMBG-2.0"
logger.info(f"Available models: {available_models}")


def process_with_rmbg(image):
    """Process image with RMBG-2.0 model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Preprocess
    input_tensor = rmbg_transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        preds = rmbg_model(input_tensor)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    
    # Convert to PIL mask
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    
    # Apply mask to image
    result = image.copy()
    result.putalpha(mask)
    
    return result

# Create a global lock for GPU operations
gpu_lock = asyncio.Lock()

@app.post("/remove_background/")
async def remove_background(file: UploadFile = File(...), method: str = Form(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        start_time = time.time()

        async def process_image():
            if method in ['tracer', 'basnet']:
                # Use CPU for carvekit models (no GPU required)
                result = await asyncio.to_thread(carvekit_models[method], [image])
                return result[0]
            elif method == 'rmbg' and rmbg_available:
                # Use RMBG-2.0 model
                result = await asyncio.to_thread(process_with_rmbg, image)
                return result
            else:
                available = "tracer, basnet"
                if rmbg_available:
                    available += ", rmbg"
                raise HTTPException(status_code=400, detail=f"Method '{method}' not available. Available methods: {available}")

        no_bg_image = await process_image()
        
        process_time = time.time() - start_time
        print(f"Background removal time ({method}): {process_time:.2f} seconds")
        
        with io.BytesIO() as output:
            no_bg_image.save(output, format="PNG")
            content = output.getvalue()

        return Response(content=content, media_type="image/png")

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def process_frame(frame_path, method):
    img = Image.open(frame_path).convert('RGB')
    
    if method in ['tracer', 'basnet']:
        # For video processing, we'll handle carvekit models in the main loop
        raise ValueError(f"Method '{method}' should be processed in the batch loop")
    elif method == 'rmbg' and rmbg_available:
        # Process with RMBG-2.0
        processed_frame = await asyncio.to_thread(process_with_rmbg, img)
        return processed_frame
    else:
        available = "tracer, basnet"
        if rmbg_available:
            available += ", rmbg"
        raise ValueError(f"Method '{method}' not available. Available methods: {available}")
    
    return None

async def process_video(video_path, method, video_id):
    try:
        processing_status[video_id] = {'status': 'processing', 'progress': 0, 'message': 'Initializing'}
        
        logger.info(f"Starting video processing: {video_path}")
        logger.info(f"Method: {method}")
        logger.info(f"Video ID: {video_id}")


        # Check video frame count
        frame_count_command = ['ffmpeg.ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', 
                               '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', video_path]
        process = await asyncio.create_subprocess_exec(
            *frame_count_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error counting frames: {stderr.decode()}")
            processing_status[video_id] = {'status': 'error', 'message': 'Error counting frames'}
            return

        frame_count = int(stdout.decode().strip())
        logger.info(f"Video frame count: {frame_count}")

        #DISABLED VIDEO LENGTH LIMIT
        #if frame_count > 250:
        #    logger.warning(f"Video too long: {frame_count} frames")
        #    processing_status[video_id] = {'status': 'error', 'message': 'Video too long (max 250 frames)'}
        #    return

        # Create a unique directory for this video's frames
        frames_dir = os.path.join(FRAMES_DIR, video_id)
        os.makedirs(frames_dir, exist_ok=True)
        logger.info(f"Created frames directory: {frames_dir}")

        # Extract frames from video
        processing_status[video_id] = {'status': 'processing', 'progress': 0, 'message': 'Extracting frames'}
        extract_command = ['ffmpeg', '-i', video_path, f'{frames_dir}/frame_%05d.png']
        logger.info(f"Executing frame extraction command: {' '.join(extract_command)}")
        process = await asyncio.create_subprocess_exec(
            *extract_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error extracting frames: {stderr.decode()}")
            processing_status[video_id] = {'status': 'error', 'message': 'Error extracting frames'}
            return

        # Process frames
        processing_status[video_id] = {'status': 'processing', 'progress': 0, 'message': 'Removing background'}
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        total_frames = len(frame_files)
        logger.info(f"Number of extracted frames: {total_frames}")

        if total_frames == 0:
            logger.error("No frames were extracted from the video")
            processing_status[video_id] = {'status': 'error', 'message': 'No frames were extracted from the video'}
            return

        # Process frames with available models
        if method in ['tracer', 'basnet']:
            # Use carvekit model for batch processing (CPU only)
            @contextmanager
            def carvekit_video_model_context(model_name):
                model = carvekit_models[model_name]
                yield model
            
            with carvekit_video_model_context(method) as model:
                async def process_frame_batch(start_idx, end_idx):
                    for i in range(start_idx, min(end_idx, total_frames)):
                        frame_file = frame_files[i]
                        frame_path = os.path.join(frames_dir, frame_file)
                        img = Image.open(frame_path).convert('RGB')
                        processed_frame = await asyncio.to_thread(model, [img])
                        processed_frame[0].save(frame_path, format='PNG')
                        progress = (i + 1) / total_frames * 100
                        processing_status[video_id] = {'status': 'processing', 'progress': progress}
                
                batch_size = 3
                for i in range(0, total_frames, batch_size):
                    await process_frame_batch(i, i + batch_size)
                    await asyncio.sleep(0)  # Allow other tasks to run
        elif method == 'rmbg' and rmbg_available:
            # Use RMBG-2.0 for video processing
            async def process_frame_batch(start_idx, end_idx):
                for i in range(start_idx, min(end_idx, total_frames)):
                    frame_file = frame_files[i]
                    frame_path = os.path.join(frames_dir, frame_file)
                    processed_frame = await process_frame(frame_path, method)
                    processed_frame.save(frame_path, format='PNG')
                    progress = (i + 1) / total_frames * 100
                    processing_status[video_id] = {'status': 'processing', 'progress': progress}
            
            batch_size = 3
            for i in range(0, total_frames, batch_size):
                await process_frame_batch(i, i + batch_size)
                await asyncio.sleep(0)  # Allow other tasks to run
        else:
            available = "tracer, basnet"
            if rmbg_available:
                available += ", rmbg"
            raise ValueError(f"Method '{method}' not available for video processing. Available methods: {available}")

        # Create output video
        processing_status[video_id] = {'status': 'processing', 'progress': 100, 'message': 'Encoding video'}
        output_path = os.path.join(TEMP_VIDEOS_DIR, f"output_{video_id}.webm")
        create_video_command = [
            'ffmpeg',
            '-framerate', '24',
            '-i', f'{frames_dir}/frame_%05d.png',
            '-c:v', 'libvpx-vp9',
            '-pix_fmt', 'yuva420p',
            '-lossless', '1',
            output_path
        ]
        logger.info(f"Executing video creation command: {' '.join(create_video_command)}")
        process = await asyncio.create_subprocess_exec(
            *create_video_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error creating output video: {stderr.decode()}")
            processing_status[video_id] = {'status': 'error', 'message': 'Error creating output video'}
            return

        logger.info(f"Video processing completed. Output path: {output_path}")
        processing_status[video_id] = {'status': 'completed', 'output_path': output_path}

    except Exception as e:
        logger.exception("Error in video processing")
        processing_status[video_id] = {'status': 'error', 'message': str(e)}
    finally:
        torch.cuda.empty_cache()

        # Clean up frames directory
        for file in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, file))
        os.rmdir(frames_dir)
        logger.info(f"Cleaned up frames directory: {frames_dir}")

@app.post("/remove_background_video/")
async def remove_background_video(background_tasks: BackgroundTasks, file: UploadFile = File(...), method: str = Form(...)):
    try:
        logger.info(f"Starting video background removal with method: {method}")
        
        # Generate a unique filename for the uploaded video
        video_id = str(uuid.uuid4())
        filename = f"input_{video_id}.mp4"
        file_path = os.path.join(TEMP_VIDEOS_DIR, filename)
        
        # Save uploaded video to the temp_videos folder
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"Video file saved: {file_path}")
        logger.info(f"File exists: {os.path.exists(file_path)}")
        logger.info(f"File size: {os.path.getsize(file_path)} bytes")

        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail=f"Failed to create video file: {file_path}")

        # Start processing in the background
        background_tasks.add_task(process_video, file_path, method, video_id)
        
        return {"video_id": video_id}

    except Exception as e:
        logger.exception(f"Error in video processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in video processing: {str(e)}")

@app.get("/status/{video_id}")
async def get_status(video_id: str):
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video ID not found")
    
    status = processing_status[video_id]
    
    if status['status'] == 'completed':
        output_path = status['output_path']
        if not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail="Processed video file not found")
        
        return FileResponse(output_path, media_type="video/webm", filename=f"processed_video_{video_id}.webm")
    
    return status

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_old_videos())
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9876)