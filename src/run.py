from lentille import Resnet50Embedder

embedder = Resnet50Embedder.from_file(
    "downloaded_models/resnet50/resnet50-v1-12-int8.onnx"
)
image_location = "test_data/cat.jpg"
# img = Image.open("test_data/dog.jpg")
# img = Image.open("test_data/segmented_airplane.png")

result = embedder.classify(image_location)

print(np.argmax(result))
