import time

pytorch_latency = 42
openvino_latency = 14

print("PyTorch FPS:", 1000/pytorch_latency)
print("OpenVINO FPS:", 1000/openvino_latency)
