import argparse
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
args = ap.parse_args()

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

image = Image.open(args.image)
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)

print("VLM Description:")
print(processor.decode(out[0], skip_special_tokens=True))
