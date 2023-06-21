import io

from fastapi import FastAPI, File
from starlette.responses import Response

from segmentation_module import get_model, get_segmentation_map

model = get_model()
app = FastAPI(title="SwinTransformer Segmentation")


@app.get("/")
def main():
    return "Hello World"


@app.post("/segmentation")
def extract_segmentation_map(file: bytes = File(...)):
    segmentation_map = get_segmentation_map(model, file, 512)
    bytes_io = io.BytesIO()
    segmentation_map.save(bytes_io, format="PNG")
    return Response(bytes_io.getvalue(), media_type="image/png")
