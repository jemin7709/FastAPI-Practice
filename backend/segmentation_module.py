import io

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image


def label2rgb(label):
    PALETTE = [
        (220, 20, 60),
        (119, 11, 32),
        (0, 0, 142),
        (0, 0, 230),
        (106, 0, 228),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 70),
        (0, 0, 192),
        (250, 170, 30),
        (100, 170, 30),
        (220, 220, 0),
        (175, 116, 175),
        (250, 0, 30),
        (165, 42, 42),
        (255, 77, 255),
        (0, 226, 252),
        (182, 182, 255),
        (0, 82, 0),
        (120, 166, 157),
        (110, 76, 0),
        (174, 57, 255),
        (199, 100, 0),
        (72, 0, 118),
        (255, 179, 240),
        (0, 125, 92),
        (209, 0, 151),
        (188, 208, 182),
        (0, 220, 176),
    ]

    image_size = label.shape[1:] + (3,)
    image = np.zeros(image_size, dtype=np.uint8)
    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]

    return image


def get_model():
    model = torch.jit.load("../models/swin-fpn.pt", map_location="cpu")
    model.eval()
    return model


def get_segmentation_map(model, binary_image, max_size) -> Image:
    image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    width, height = image.size
    resize_factor = min(max_size / width, max_size / height)
    resized_image = (
        np.array(
            image.resize(
                (
                    int(image.width * resize_factor),
                    int(image.height * resize_factor),
                )
            )
        )
        / 255.0
    )
    preprocess = A.Compose([ToTensorV2()])
    input: torch.Tensor = preprocess(**{"image": resized_image})["image"]
    input: torch.Tensor = input.unsqueeze(0).float()
    with torch.no_grad():
        output = model(input)
    output = F.interpolate(output, size=(width, height), mode="bilinear")
    output = torch.sigmoid(output)
    output = (output > 0.5).detach().numpy()
    output = Image.fromarray(label2rgb(output[0]))
    return output
