import logging
import os
import pathlib
import shutil
import time
from pathlib import Path

import clip
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms

log_file_name = f"bullseye.log"
time_format = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt=time_format
)
logging.basicConfig(filename=log_file_name)

logging.info("Started to load model CLIP")
model_load_start = time.perf_counter()


# Load CLIP model (automatically uses CPU if no GPU is available)
def load_model(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


# if pathlib.Path(os.getcwd(), "clip_image_encoder.onnx").exists() is False:
def create_ort_session(model, device):
    onnx_path = "clip_image_encoder.onnx"
    if pathlib.Path(os.getcwd(), onnx_path).exists():
        return ort.InferenceSession(onnx_path)
    else:
        dummy_image_input = torch.randn(1, 3, 224, 224).to(
            device
        )  # [batch, channels, height, width]
        torch.onnx.export(
            model.visual,  # Only export image encoder
            dummy_image_input,
            "clip_image_encoder.onnx",
            input_names=["image"],
            output_names=["features"],
            dynamic_axes={
                "image": {0: "batch"},  # Dynamic batch size
                "features": {0: "batch"},
            },
            opset_version=14,  # ONNX opset (must be ≥14 for CLIP)
        )
        return ort.InferenceSession(onnx_path)


# model, preprocess = clip.load("ViT-B/32", device=device)
# model.eval()
# model_load_finish = time.perf_counter()
# logging.info(f"Model CLIP loaded in {model_load_finish-model_load_start :0.2f}")
# if pathlib.Path(os.getcwd(), "clip_image_encoder.onnx").exists() is False:
#     logging.info("No ONNX model found")
#     # Create a dummy input matching CLIP's expected format
#     dummy_image_input = torch.randn(1, 3, 224, 224).to(
#         device
#     )  # [batch, channels, height, width]
#     torch.onnx.export(
#         model.visual,  # Only export image encoder
#         dummy_image_input,
#         "clip_image_encoder.onnx",
#         input_names=["image"],
#         output_names=["features"],
#         dynamic_axes={
#             "image": {0: "batch"},  # Dynamic batch size
#             "features": {0: "batch"},
#         },
#         opset_version=14,  # ONNX opset (must be ≥14 for CLIP)
#     )
#     logging.info("ONNX model creating")
# ort_session = ort.InferenceSession("clip_image_encoder.onnx")
# logging.info("ONNX model loaded")


def preprocess_image(image_path):
    """Preprocess an image for CLIP model input"""
    # Load image
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Define the transformation pipeline
    transform_pipeline = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    # Apply transformations
    image_tensor = transform_pipeline(image).unsqueeze(0)  # Add batch dimension

    # Convert to numpy array for ONNX Runtime
    return image_tensor.detach().cpu().numpy()


def process_images(dir_strpath: str, labels, device, model, ort_session):
    dir = Path(dir_strpath)
    # Your custom labels (add/remove as needed)
    if dir.exists() is False:
        raise OSError(f"Directory does not exist: {dir_strpath}")
    else:
        # sys.stderr("Error reading file")
        text = clip.tokenize(labels).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        for file in os.listdir(dir):
            img_path = dir.joinpath(file)
            try:
                image = preprocess_image(img_path)

                # Inference
                image_features = ort_session.run(None, {"image": image})[0]
                image_features = torch.from_numpy(image_features).to(device)

                # Normalize features (critical!)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # Similarity with temperature scaling
                similarity = (
                    image_features @ text_features.T
                ) * model.logit_scale.exp()
                probs = similarity.softmax(dim=-1)
                highest_match_label = labels[probs[0].argmax()]
                subfolder_label = dir.joinpath(highest_match_label)
                if pathlib.Path(subfolder_label).is_dir():
                    shutil.move(
                        str(dir.joinpath(file)), str(subfolder_label.joinpath(file))
                    )
                else:
                    os.makedirs(subfolder_label, exist_ok=True)  # Better than os.mkdir
                    shutil.move(
                        str(dir.joinpath(file)), str(subfolder_label.joinpath(file))
                    )
                logging.info(f"{file} processed succesfully")
                print("Classification results:")
                for label, prob in zip(labels, probs[0]):
                    print(f"{file} {label}: {prob:.2%}")
                print(f"Most likely match: {highest_match_label}")
            except Exception as e:
                print(f"Error processing {file}: {e}")
                logging.error(f"{file} processed unsuccesfully")
