import argparse,os,sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from inference.infer_pytorch import run_pytorch
from inference.infer_openvino import run_openvino
import time
ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--device", default="CPU")
args = ap.parse_args()

_, _, pt = run_pytorch(args.image)
_, _, ov = run_openvino(args.image, args.device)

print("\nFramework   Latency (ms)")
print("------------------------")
print(f"PyTorch     {pt:.2f}")
print(f"OpenVINO    {ov:.2f}")
print(f"Speedup     {pt/ov:.2f}x")
