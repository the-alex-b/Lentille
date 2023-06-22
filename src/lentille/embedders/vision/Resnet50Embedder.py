import numpy as np
from PIL import Image

from lentille.base_embedder import BaseEmbedder
from lentille.utils import resnet50_preprocess


class Resnet50Embedder(BaseEmbedder):
    """
    Always instantiate using: Resnet50Embedder.from_file(model_file.onnx)
    Models can be found here: https://github.com/onnx/models/tree/main/vision/classification/resnet
    """

    def _preprocess(self, image):
        return resnet50_preprocess.preprocess(image)

    def classify(
        self, image_location: str = None, image_bytes: Image = None
    ) -> np.array:
        if image_location and image_bytes:
            raise Exception(
                "Please provide an image location or image bytes, not both."
            )

        if image_location:
            image = Image.open(image_location)
            preprocessed_image = self._preprocess(image)

        if image_bytes:
            preprocessed_image = self._preprocess(image_bytes)

        # Magic structure to be able to speak to onnx?
        input_name = self.session.get_inputs()[0].name
        input_data = {input_name: preprocessed_image}

        # Label overview: https://files.fast.ai/models/imagenet_class_index.json
        return self.session.run([], input_data)[0]
