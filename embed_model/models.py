from typing import Any, List

from llama_index.core.embeddings import BaseEmbedding
from sentence_transformers import SentenceTransformer


class LocalHuggingFaceModel(BaseEmbedding):
    def __init__(
            self,
            instructor_model_name: str = r"E:\pretrain_model\text_embedding\bge_large_zh",  # 本地HuggingFace模型的路径
            instruction: str = "Represent the Computer Science documentation or question:",
            **kwargs: Any,
    ) -> None:
        self._model = SentenceTransformer(model_name_or_path=instructor_model_name)  #
        self._instruction = instruction
        super().__init__(**kwargs)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode(query).tolist()  # [[self._instruction, query]]
        return embeddings

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode(text).tolist()
        return embeddings

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode([text for text in texts]).tolist()
        return embeddings

    async def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str):
        pass


