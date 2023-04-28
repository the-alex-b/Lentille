from lentille.base_embedder import BaseEmbedder
from lentille.utils import resnet50_preprocess
import numpy as np
import onnxruntime


# Models can be found here: https://github.com/onnx/models/tree/main/vision/classification/resnet
class Resnet50Embedder(BaseEmbedder):
    def __init__(self):
        # Load the ONNX model from disk
        self.session = onnxruntime.InferenceSession(
            "src/lentille/weights/resnet50/resnet50-v1-12-int8.onnx"
        )

    def _preprocess(self, image):
        return resnet50_preprocess.preprocess(image)

    def classify(self, image):
        preprocessed_image = self._preprocess(image)

        # Magic structure to be able to speak to onnx?
        input_name = self.session.get_inputs()[0].name
        input_data = {input_name: preprocessed_image}

        return self.session.run([], input_data)[0]
