from abc import ABC
import onnxruntime


class BaseEmbedder(ABC):
    def __init__(self, session):
        self.session = session

    @classmethod
    def from_file(cls, file):
        return cls(onnxruntime.InferenceSession(file))
