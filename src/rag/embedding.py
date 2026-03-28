"""Qwen3-VL Multimodal embedding model, conforming to LangChain Embeddings interface.

Support mixed input（text + image），Uniformly passed in str or dict format。
"""

import asyncio
from typing import Union

from langchain_core.embeddings import Embeddings

from src.custom.qwen3_vl_embedding import Qwen3VLEmbedder


class Qwen3VLEmbeddings(Embeddings):
    """Qwen3-VL Multimodal embedding model, conforming to LangChain Embeddings interface.

    Support mixed input（text + image），Uniformly passed in str or dict format。

    Examples:
        Plain text embedding：
            embed_query("What is machine learning？")
            embed_documents(["text1", "text2"])

        Image embedding：
            embed_query({"image": "/path/to/image.jpg"})
            embed_documents([{"image": "/path/to/img1.jpg"}, {"image": "/path/to/img2.jpg"}])

        hybrid embedding：
            embed_query({"text": "Description text", "image": "/path/to/image.jpg"})
            embed_documents([
                {"text": "plain text"},
                {"image": "/path/to/image.jpg"},
                {"text": "Mixed graphics and text", "image": "/path/to/another.jpg"}
            ])
    """

    def __init__(
        self,
        model_name_or_path: str = "../models/Qwen3-VL-Embedding-2B",
        **kwargs,
    ):
        """Initialize the Qwen3-VL embedding model。

        Args:
            model_name_or_path: Model path or HuggingFace model ID
            **kwargs: Additional parameters passed to Qwen3VLEmbedder
        """
        self.model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path, **kwargs)

    def _normalize_input(self, input: Union[str, dict]) -> dict:
        """Normalize input to the dict format required by Qwen3VL。

        Args:
            input: Plain text string or formatted dict

        Returns:
            Standardized dict, including text, image and other keys
        """
        if isinstance(input, str):
            return {"text": input}
        return input

    def embed_documents(
        self,
        inputs: list[Union[str, dict]],
        instruction: str = None,
    ) -> list[list[float]]:
        """Batch embedded documents (supports mixed input）。

        Args:
            inputs: List of strings or dicts, dict format：
                - {"text": "description text"}
                - {"image": "/path/to/image.jpg"}
                - {"text": "describe", "image": "/path/to/image.jpg"}
                - {"text": "...", "image": "...", "video": "..."}  (Support video)
            instruction: Optional embedded directive, applied to all inputs

        Returns:
            A list of embedding vectors, each vector being list[float]
        """
        normalized = [self._normalize_input(inp) for inp in inputs]
        if instruction:
            for item in normalized:
                item["instruction"] = instruction
        embeddings = self.model.process(normalized)
        return embeddings.tolist()

    def embed_query(
        self,
        input: Union[str, dict],
        instruction: str = "Retrieve images or text relevant to the user's query.",
    ) -> list[float]:
        """Embed a single query (supports mixed input）。

        Args:
            input: plain text string, or dict format：
                - {"text": "description text"}
                - {"image": "/path/to/image.jpg"}
                - {"text": "describe", "image": "/path/to/image.jpg"}
                - {"text": "...", "image": "...", "video": "..."}  (Support video)
            instruction: Embed command, default is retrieval command

        Returns:
            embedding vector list[float]
        """
        normalized = self._normalize_input(input)
        normalized["instruction"] = instruction
        embeddings = self.model.process([normalized])
        return embeddings.tolist()[0]

    async def aembed_documents(
        self,
        inputs: list[Union[str, dict]],
        instruction: str = None,
    ) -> list[list[float]]:
        """Asynchronous version of embed_documents.

        Use a thread pool to perform synchronous embedded operations to avoid blocking the event loop。
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.embed_documents, inputs, instruction
        )

    async def aembed_query(self, input: Union[str, dict]) -> list[float]:
        """Asynchronous version of embed_query.

        Use a thread pool to perform synchronous embedded operations to avoid blocking the event loop。
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, input)
