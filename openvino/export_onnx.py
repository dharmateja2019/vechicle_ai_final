import torch

import sys
import os
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from models.detection_model import VehicleDetector
model = VehicleDetector()
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "vehicle_detector.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

dummy = torch.randn(1, 3, 320, 320)

torch.onnx.export(
    model,
    dummy,
    "vehicle_detector.onnx",   # let PyTorch choose opset
    export_params=True,
    do_constant_folding=True
)

print("✅ ONNX export successful (opset 18)")
