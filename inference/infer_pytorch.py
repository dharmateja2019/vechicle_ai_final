import torch, cv2
from models.detection_model import VehicleDetector
from models.recognition_model import VehicleRecognition

TYPE = ["Car", "Bus", "Truck", "Bike"]
COLOR = ["White", "Black", "Red", "Blue", "Gray", "Yellow"]

detector = VehicleDetector()
detector.load_state_dict(torch.load("vehicle_detector.pth"))
detector.eval()

recognizer = VehicleRecognition()
recognizer.eval()

img = cv2.imread("data/images/car1.jpg")
x = torch.tensor(img).permute(2,0,1).unsqueeze(0)/255

with torch.no_grad():
    cls, box = detector(x)
    vehicle_type = TYPE[cls.argmax().item()]

    x1,y1,w,h = box[0].int()
    crop = img[y1:y1+h, x1:x1+w]
    crop = torch.tensor(crop).permute(2,0,1).unsqueeze(0)/255

    t, c = recognizer(crop)
    color = COLOR[c.argmax().item()]

cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,255,0),2)
cv2.putText(img,f"{vehicle_type} | {color}",(x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

cv2.imwrite("output.jpg", img)
print("Output saved as output.jpg")
