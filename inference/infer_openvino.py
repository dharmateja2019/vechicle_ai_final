from openvino.runtime import Core
import cv2, time

ie = Core()
model = ie.read_model("vehicle_detector.xml")
compiled = ie.compile_model(model, "CPU")
infer = compiled.create_infer_request()

img = cv2.imread("test.jpg")
blob = cv2.resize(img,(320,320)).transpose(2,0,1)[None]/255

start = time.time()
infer.infer({0: blob})
latency = (time.time()-start)*1000

print(f"OpenVINO latency: {latency:.2f} ms")
