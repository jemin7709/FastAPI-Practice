import io

import requests
import streamlit as st
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

# interact with FastAPI endpoint
# backend = "http://fastapi:8000/segmentation"
backend = "http://127.0.0.1:8000/segmentation"


def process(image, server_url: str):
    m = MultipartEncoder(fields={"file": ("filename", image, "image/png")})

    r = requests.post(server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000)

    return r


# construct UI layout
st.title("Hand Bone Segmentation")

input_image = st.file_uploader("insert image")  # image upload widget

if st.button("Get segmentation map"):
    col1, col2 = st.columns(2)

    if input_image:
        segments = process(input_image, backend)
        original_image = Image.open(input_image).convert("RGB")
        segmented_image = Image.open(io.BytesIO(segments.content)).convert("RGB")
        col1.header("Original")
        col1.image(original_image, use_column_width=True)
        col2.header("Segmented")
        col2.image(segmented_image, use_column_width=True)
    else:
        # handle case with no image
        st.write("Insert an image!")
