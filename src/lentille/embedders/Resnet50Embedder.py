from lentille.base_embedder import BaseEmbedder
from lentille.utils import resnet50_preprocess
import numpy as np


# Models can be found here: https://github.com/onnx/models/tree/main/vision/classification/resnet
class Resnet50Embedder(BaseEmbedder):
    def _preprocess(self, image):
        return resnet50_preprocess.preprocess(image)

    def classify(self, image):
        preprocessed_image = self._preprocess(image)

        # Magic structure to be able to speak to onnx?
        input_name = self.session.get_inputs()[0].name
        input_data = {input_name: preprocessed_image}

        return self.session.run([], input_data)[0]
