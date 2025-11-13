import os
import cv2
import time
import torch
import subprocess
from PIL import Image
from cog import BasePredictor, Input, Path
from typing import Dict, Any, Optional, List
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_PATH = "./checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/Qwen/Qwen2-VL-8B-Instruct/model.tar"

def download_weights(url, dest):
    start = time.time()
    print(f"Downloading model from {url} ...")
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("Download completed in", round(time.time() - start, 2), "s")

def extract_frames(video_path: str, num_frames: int = 10) -> List[Image.Image]:
    """Extract evenly spaced frames from a video file."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames == 0:
        return []

    for i in range(num_frames):
        frame_idx = int(i * total_frames / num_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load model and processor."""
        if not os.path.exists(MODEL_PATH):
            download_weights(MODEL_URL, MODEL_PATH)

        self.model_id = "checkpoints"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto",
        ).eval()

    def predict(
        self,
        video: Path = Input(description="Video file (mp4, mov, avi, etc.)", default=None),
        system_message: str = Input(description="System instruction or background context", default=""),
        user_message: str = Input(description="User query or prompt", default="Describe what is happening."),
        num_frames: int = Input(description="Number of frames to sample from the video", default=10, ge=1, le=30),
        max_new_tokens: int = Input(description="Maximum output tokens", default=512),
        temperature: float = Input(description="Sampling temperature", default=0.7, ge=0.0, le=2.0),
        top_p: float = Input(description="Top-p (nucleus) sampling", default=0.9, ge=0.0, le=1.0),
    ) -> str:
        """Run inference with video + system/user messages."""

        if video is None:
            return "Error: No video provided."

        # Extract frames
        frames = extract_frames(str(video), num_frames=num_frames)
        if not frames:
            return "Error: Could not extract frames from the video."

        # Build the messages list
        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_message}]
                          + [{"type": "image", "image": img} for img in frames],
            },
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process all vision data
        image_inputs, video_inputs = process_vision_info(messages)
        batch = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate output
        with torch.inference_mode():
            output_ids = self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
            )

        # Decode only the new tokens
        prompt_len = batch["input_ids"].shape[-1]
        trimmed = output_ids[:, prompt_len:]
        text_out = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return text_out.strip()
