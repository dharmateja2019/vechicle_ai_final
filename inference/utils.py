import cv2
import numpy as np
import torch

# --------------------------------------------------
# Safe crop using absolute pixel coordinates
# --------------------------------------------------
def safe_crop(img, x1, y1, x2, y2, min_size=10):
    """
    Safely crop an image using pixel coordinates.
    Returns None if crop is invalid or too small.
    """
    h, w, _ = img.shape

    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))

    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return None

    crop = img[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None

    return crop


# --------------------------------------------------
# Center crop (used to avoid background influence)
# --------------------------------------------------
def center_crop(img, x1, y1, x2, y2, ratio=0.6):
    """
    Crop central region of a bounding box.
    """
    h, w, _ = img.shape

    bw = x2 - x1
    bh = y2 - y1

    cx1 = int(x1 + (1 - ratio) * bw / 2)
    cy1 = int(y1 + (1 - ratio) * bh / 2)
    cx2 = int(x2 - (1 - ratio) * bw / 2)
    cy2 = int(y2 - (1 - ratio) * bh / 2)

    cx1 = max(0, min(cx1, w - 1))
    cy1 = max(0, min(cy1, h - 1))
    cx2 = max(0, min(cx2, w))
    cy2 = max(0, min(cy2, h))

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    crop = img[cy1:cy2, cx1:cx2]
    if crop is None or crop.size == 0:
        return None

    return crop


# --------------------------------------------------
# Preprocess crop for CNN (if needed later)
# --------------------------------------------------
def preprocess_crop(crop):
    """
    Resize + normalize for MobileNet-based models
    """
    crop = cv2.resize(crop, (224, 224))
    crop_t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    crop_t = (crop_t - 0.5) / 0.5
    return crop_t


# --------------------------------------------------
# Robust HSV + BGR vehicle color detection
# --------------------------------------------------
import cv2
import numpy as np

def detect_vehicle_color(crop):
    """
    Deterministic vehicle color detection.
    NEVER returns Unknown if crop is valid.
    """

    if crop is None or crop.size == 0:
        return "Unknown"

    # Resize for stability
    crop = cv2.resize(crop, (64, 64))

    # Mean BGR values
    b, g, r = crop.mean(axis=(0, 1))
    brightness = (r + g + b) / 3

    # --- Brightness based ---
    if brightness < 70:
        return "Black"

    if brightness > 200:
        return "White"

    # --- Dominant color channel ---
    if r > g + 15 and r > b + 15:
        return "Red"

    if b > r + 15 and b > g + 15:
        return "Blue"

    if g > r + 15 and g > b + 15:
        return "Green"

    # --- Remaining cases ---
    return "Gray"
