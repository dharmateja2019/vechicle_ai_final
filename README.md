# Vehicle Detection & Recognition using MobileNetV3

## Overview

This project implements an end-to-end **Vehicle Detection and Recognition** pipeline
using **MobileNetV3**. The system is optimized for **CPU and edge environments** using
**OpenVINO Runtime** with **ONNX models**, enabling fast and efficient inference.

The pipeline detects vehicles from images, identifies their **type** (Car, Bus, Truck, Bike),
and predicts their **color**, while providing performance comparisons between
**PyTorch** and **OpenVINO**.

---

## Key Features

- Lightweight **MobileNetV3** backbone
- Vehicle detection with bounding boxes
- Vehicle recognition (type + color)
- PyTorch → ONNX → OpenVINO Runtime pipeline
- CPU-optimized inference
- Performance benchmarking
- Modular and scalable architecture

---

## Supported Vehicle Types

- Car
- Bus
- Truck
- Bike

## Supported Vehicle Colors

- White
- Black
- Red
- Blue
- Gray
- Yellow

---

## Project Pipeline

Input Image
↓
Vehicle Detection (MobileNetV3 – PyTorch)
↓
ONNX Export
↓
OpenVINO Runtime Inference
↓
Vehicle Crop
↓
Vehicle Recognition (Type + Color)
↓
Final Annotated Output

yaml
Copy code

---

## Directory Structure

vehicle_ai_final/
│
├── data/
│ ├── images/
│ └── annotations/
│
├── models/
│ ├── detection_model.py
│ └── recognition_model.py and .pth,.onnx files
│
├── training/
│ └── create_dummy_checkpoint.py,
│
├── inference/
│ ├── infer_openvino.py
│ └── labels.py, infer_pytorch.py, infer_vlm.py, utils.py, compare_inference.py
│
├── openvino/
│ └── export_onnx.py
│
├── gui/
│ └── app.py
│
├── functional_workflow.yaml
├── project_architecture.yaml
└── README.md

yaml
Copy code

---

## Install dependencies

python3 -m venv vechicle_ai_env
source vechicle_ai_env/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

## Execution Steps

### 1. Create Dummy Checkpoint

```bash
python3 training/create_dummy_checkpoint.py
2. Export ONNX
python3 openvino/export_onnx.py

3. Run OpenVINO Inference
python3 inference/infer_openvino.py

4. Performance Comparison
python3 inferenc/compare_inference.py

5. GUI
streamlit run gui/app.py
```
