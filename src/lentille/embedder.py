import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


class Embedder:
    @staticmethod
    def embed(text):
        return openai.Embedding.create(model="text-embedding-ada-002", input=text)[
            "data"
        ][0]["embedding"]
