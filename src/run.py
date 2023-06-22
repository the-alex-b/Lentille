from lentille import Resnet50Embedder
import numpy as np
from PIL import Image

embedder = Resnet50Embedder.from_file(
    "downloaded_models/resnet50/resnet50-v1-12-int8.onnx"
)
image_location = "test_data/cat.jpg"
# img = Image.open("test_data/dog.jpg")
# img = Image.open("test_data/segmented_airplane.png")

img = Image.open(image_location)

result = embedder.classify(image_bytes=img)

print(np.argmax(result))
