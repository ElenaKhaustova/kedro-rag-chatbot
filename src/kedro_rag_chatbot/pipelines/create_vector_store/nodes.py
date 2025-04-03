import logging
from typing import Any, Callable

import numpy as np
from deeplake.core.vectorstore import VectorStore
from sklearn.feature_extraction.text import HashingVectorizer

logger = logging.getLogger(__name__)


def get_texts(dialog: dict[str, Any]) -> str:
    lines = []
    if "user" in dialog and "text" in dialog:
        lines.append(f"{dialog['user']}: {dialog['text']}")
    if "replies" in dialog:
        for reply in dialog["replies"]:
            if "user" in reply and "text" in reply:
                lines.append(f"{reply['user']}: {reply['text']}")
    return "\n\n".join(lines)


def format_dialogs(dialogs: dict[str, dict]) -> dict[str, str]:
    texts = {}
    for dialog_name, dialog in dialogs.items():
        text = get_texts(dialog)
        if text:
            texts[dialog_name] = text
            logger.info(f"Formatted dialog {dialog_name}")
        else:
            logger.warning(f"Empty dialog {dialog_name}")
    return texts


def create_embedding_function() -> Callable:
    def embedding_function(texts: list[str], embeddings_size: int = 2**16) -> list:
        if isinstance(texts, str):
            texts = [texts]
        vectorizer = HashingVectorizer(n_features=embeddings_size, dtype=np.float32)
        X = vectorizer.fit_transform(texts)

        return [vector.toarray().flatten() for vector in X]

    return embedding_function


def create_vector_store(
    vector_store: VectorStore,
    formatted_dialogs: dict[str, str],
    embedding_function: Callable,
    embeddings_size: int,
) -> VectorStore:
    texts = [dialog for dialog in formatted_dialogs.values()]
    embeddings = embedding_function(texts, embeddings_size)
    vector_store.add(
        text=texts,
        embedding=embeddings,
        metadata=[{"dialog_id": dialog_id} for dialog_id in formatted_dialogs.keys()],
    )

    return vector_store
