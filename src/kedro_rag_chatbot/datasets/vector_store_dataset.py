from typing import Any, NoReturn

from deeplake.core.vectorstore import VectorStore
from kedro.io import AbstractDataset, DatasetError


class DeeplakeVectorStoreDataset(AbstractDataset[None, VectorStore]):
    """Kedro dataset for working with Deep Lake Vector Store

    https://docs.activeloop.ai/examples/rag/tutorials/vector-store-basics
    """

    def __init__(self, path: str, **kwargs):
        self._path = path
        self.kwargs = kwargs or {}

    def load(self) -> VectorStore:
        return VectorStore(path=self._path, **self.kwargs)

    def save(self, data: None) -> NoReturn:
        raise DatasetError(f"{self.__class__.__name__} is a read only dataset type")

    def _describe(self) -> dict[str, Any]:
        return {"filepath": self._path, **self.kwargs}
