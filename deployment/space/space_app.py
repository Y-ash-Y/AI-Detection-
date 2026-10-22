"""Space-only entrypoint: serve the explicitly released bundle, never a fallback."""
import os
from pathlib import Path

# Prevent old Space variables from silently selecting retired checkpoints.
for key in ("PMSA_CKPT", "PMSA_CAL", "PMSA_ALLOW_LEGACY", "PMSA_TRUSTED_LEGACY",
            "PMSA_LEGACY_NORMALIZE_JPEG"):
    os.environ.pop(key, None)
os.environ["PMSA_BUNDLE"] = str(Path(__file__).resolve().parent / "model")
os.environ["PMSA_DEVICE"] = "cpu"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "2")

import torch
torch.set_num_threads(2)

from app import build_demo

if __name__ == "__main__":
    build_demo().launch(server_name=os.getenv("GRADIO_SERVER_NAME","0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT","7860")), max_file_size="20mb")
