from abc import ABC


class BaseEmbedder(ABC):
    def __init__(self):
        print("I am the base class")
