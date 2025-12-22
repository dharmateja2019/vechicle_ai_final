import torch, cv2, json
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from models.detection_model import VehicleDetector

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VehicleDetector().to(device)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)

with open("data/annotations/vehicles.json") as f:
    data = json.load(f)

for epoch in range(10):
    total_loss = 0
    for ann in data["annotations"]:
        img_info = next(i for i in data["images"] if i["id"] == ann["image_id"])
        img = cv2.imread(f"data/images/{img_info['file_name']}")
        x = torch.tensor(img).permute(2,0,1).unsqueeze(0)/255
        x = x.to(device)

        cls_pred, box_pred = model(x)

        cls_target = torch.tensor([ann["category_id"]]).to(device)
        box_target = torch.tensor([ann["bbox"]]).float().to(device)

        loss = (
            torch.nn.functional.cross_entropy(cls_pred, cls_target) +
            torch.nn.functional.l1_loss(box_pred, box_target)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch} | Loss {total_loss:.4f}")

torch.save(model.state_dict(), "vehicle_detector.pth")
