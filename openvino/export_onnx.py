import torch

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from models.detection_model import VehicleDetector
model = VehicleDetector()
model.load_state_dict(torch.load("vehicle_detector.pth"))
model.eval()

dummy = torch.randn(1,3,320,320)

torch.onnx.export(
    model,
    dummy,
    "vehicle_detector.onnx",
    opset_version=11
)
