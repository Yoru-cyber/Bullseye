import os
import shutil
import clip
import torch
from PIL import Image
import pathlib

# Load CLIP model (automatically uses CPU if no GPU is available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Your custom labels (add/remove as needed)
labels = [
    "superheroe",
    "anime",
]
for file in os.listdir(pathlib.Path("test")):

    img_path = pathlib.Path("test", file)
    # Load and preprocess the image
    image = preprocess(Image.open(img_path, 'r')).unsqueeze(0).to(device)
    text = clip.tokenize(labels).to(device)

    # Run inference
    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # Print results
    print("Classification results:")
    highest_match = labels[probs[0].argmax()]
    if os.path.isdir(pathlib.Path("test", highest_match)):
        shutil.move(
            pathlib.Path("test", file), pathlib.Path("test", highest_match, file)
        )
    else:
        os.mkdir(pathlib.Path("test", highest_match), 0o755)
        shutil.move(
            pathlib.Path("test", file), pathlib.Path("test", highest_match, file)
        )
    for label, prob in zip(labels, probs[0]):
        print(f"{label}: {prob:.2%}")
    print(f"Most likely match: {highest_match}")
