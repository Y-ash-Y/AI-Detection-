"""Frame extraction utilities for video processing."""
import cv2
import torch
import numpy as np
from typing import List, Tuple, Optional


class FrameExtractor:
    """
    Extracts frames from video files with optional sampling.
    """

    def __init__(self, sample_rate: int = 1):
        """
        Args:
            sample_rate: Extract every Nth frame (1 = all frames)
        """
        self.sample_rate = sample_rate

    def extract_frames(self, video_path: str) -> torch.Tensor:
        """
        Extract frames from a video file.

        Args:
            video_path: Path to video file

        Returns:
            (F, C, H, W) tensor of frames, C=3 (RGB), range [0, 1]
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.sample_rate == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)

            frame_count += 1

        cap.release()

        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")

        # Stack into tensor (F, H, W, C) then permute to (F, C, H, W)
        frames_array = np.stack(frames, axis=0)
        frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2)

        return frames_tensor

    def get_fps(self, video_path: str) -> float:
        """Get frames per second of the video."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    def get_frame_count(self, video_path: str) -> int:
        """Get total frame count of the video."""
        cap = cv2.VideoCapture(video_path)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return count
