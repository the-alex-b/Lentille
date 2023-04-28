from lentille import Resnet50Embedder
import numpy as np
from PIL import Image


embedder = Resnet50Embedder()
img = Image.open("src/test_data/dog.jpg")

result = embedder.classify(img)

print(result)
