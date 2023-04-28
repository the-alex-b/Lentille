from lentille import Resnet50Embedder
import numpy as np
from PIL import Image

# resnet labels: https://files.fast.ai/models/imagenet_class_index.json

embedder = Resnet50Embedder()
# img = Image.open("src/test_data/cat.jpg")
img = Image.open("src/test_data/dog.jpg")

result = embedder.classify(img)

print(result)

print(np.argmax(result))
