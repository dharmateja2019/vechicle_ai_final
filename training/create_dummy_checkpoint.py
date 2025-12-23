import torch
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from models.detection_model import VehicleDetector
model = VehicleDetector()
torch.save(model.state_dict(), "vehicle_detector.pth")

print("Dummy checkpoint created successfully")
