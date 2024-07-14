from PIL import Image
import requests
from vq_clip import VQCLIPModel
from transformers import CLIPProcessor

model = VQCLIPModel.from_pretrained_clip(clip_path="openai/clip-vit-large-patch14", vision_vq_adapter_path="adams-story/vq-ViT-L-14-k64-d32-ema", )

# make prediction
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

model.to('cuda')
inputs.to(model.device)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs)
codes = outputs.image_codes # the vq codes

breakpoint()