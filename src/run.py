from lentille import Resnet50Embedder
import numpy as np
from PIL import Image

# resnet labels: https://files.fast.ai/models/imagenet_class_index.json

embedder = Resnet50Embedder.from_file(
    "downloaded_models/resnet50/resnet50-v1-12-int8.onnx"
)
# img = Image.open("src/test_data/cat.jpg")
# img = Image.open("src/test_data/dog.jpg")
img = Image.open("test_data/segmented_airplane.png")

result = embedder.classify(img)

# print(result)

print(np.argmax(result))
