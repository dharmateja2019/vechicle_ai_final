import argparse, time, cv2, numpy as np, os, sys
from openvino.runtime import Core

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from inference.labels import VEHICLE_TYPES
from inference.utils import safe_crop, detect_vehicle_color, to_numpy


def run_openvino(image_path, device="CPU"):
    ie = Core()
    model = ie.read_model("models/vehicle_detector.onnx")
    compiled = ie.compile_model(model, device)

    img = cv2.imread(image_path)
    orig = img.copy()
    h, w, _ = img.shape

    inp = cv2.resize(img, (320, 320))
    blob = inp.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    start = time.time()
    outputs = compiled([blob])
    latency = (time.time() - start) * 1000

    cls = to_numpy(outputs[0])
    box = to_numpy(outputs[1])
    # ✅ SAME box semantics as OpenVINO
    bx = box[0].tolist()
    bx = box[0] if box.ndim > 1 else box
    bx = bx.tolist()

    print("RAW BOX:", bx)
    print("IMAGE SIZE:", w, h)
# TRY OPTION A: normalized x1,y1,x2,y2
    x1 = int(bx[0] * w)
    y1 = int(bx[1] * h)
    x2 = int(bx[2] * w)
    y2 = int(bx[3] * h)

    cv2.rectangle(orig, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.imwrite("DEBUG_BOX.jpg", orig)

    print("Saved DEBUG_BOX.jpg")

    if crop is None or crop.size == 0:
        color = "Unknown"
    else:
        color = detect_vehicle_color(crop)

    cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        orig,
        f"{vehicle} | {color}",
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    cv2.imwrite("output_openvino.jpg", orig)

    print(f"Detected Vehicle : {vehicle}")
    print(f"Vehicle Color    : {color}")
    print(f"OpenVINO Latency : {latency:.2f} ms")
    print("Output saved to  : output_openvino.jpg")

    return vehicle, color, latency


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="CPU")
    args = ap.parse_args()

    run_openvino(args.image, args.device)
