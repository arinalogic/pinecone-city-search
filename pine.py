"""Простой клиент для работы с векторной базой данных Pinecone."""

import os
from typing import Any

from dotenv import load_dotenv
from pinecone import Index, NotFoundError, Pinecone, PineconeError


class PineconeClient:
    """Обёртка над SDK Pinecone: подключение, запись, чтение, поиск, удаление."""

    def __init__(self) -> None:
        """Создаёт подключение к Pinecone по ключу из .env.

        Вход:
            Ничего. Ключ читается из переменной окружения PINECONE_API_KEY.

        Выход:
            None.

        Исключения:
            ValueError: если PINECONE_API_KEY не задан.
        """
        load_dotenv()
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY не найден. Добавьте его в .env")

        self.pc = Pinecone(api_key=api_key)
        self.index: Index | None = None
        self.index_name: str | None = None

    def _require_index(self) -> Index:
        """Возвращает текущий индекс или сообщает, что подключения нет.

        Выход:
            Index: объект индекса Pinecone.

        Исключения:
            RuntimeError: если connect_to_index() ещё не вызывался.
        """
        if self.index is None:
            raise RuntimeError("Сначала вызовите connect_to_index()")
        return self.index

    def connect_to_index(self, index_name: str, dimension: int) -> None:
        """Подключается к существующему индексу по имени.

        Вход:
            index_name: имя индекса в Pinecone.
            dimension: ожидаемая размерность векторов
                (например, 1536 для text-embedding-3-small).

        Выход:
            None. Индекс сохраняется в self.index.

        Исключения:
            ValueError: если индекс не найден или размерность не совпадает.
            ConnectionError: если не удалось связаться с Pinecone.
        """
        try:
            description = self.pc.describe_index(index_name)
        except NotFoundError as error:
            raise ValueError(f"Индекс '{index_name}' не найден") from error
        except PineconeError as error:
            raise ConnectionError(f"Ошибка подключения к Pinecone: {error}") from error

        if description.dimension != dimension:
            raise ValueError(
                f"Размерность индекса {description.dimension}, "
                f"а ожидалась {dimension}"
            )

        self.index = self.pc.Index(index_name)
        self.index_name = index_name

    def upsert_vectors(self, vectors: list[dict[str, Any]]) -> int:
        """Записывает новые векторы или обновляет существующие (по id).

        Вход:
            vectors: список словарей вида
                {"id": "doc1", "values": [0.1, 0.2, ...], "metadata": {...}}.
                Ключ "metadata" необязателен.

        Выход:
            int: количество записанных векторов.

        Исключения:
            ValueError: если список пуст или у вектора нет id/values.
            RuntimeError: если индекс не подключён или запрос не удался.
        """
        index = self._require_index()

        if not vectors:
            raise ValueError("Список векторов пуст")
        for vector in vectors:
            if "id" not in vector or "values" not in vector:
                raise ValueError(f"У вектора нет 'id' или 'values': {vector}")

        try:
            response = index.upsert(vectors=vectors)
        except PineconeError as error:
            raise RuntimeError(f"Ошибка записи векторов: {error}") from error

        return response.upserted_count

    def fetch(self, vector_id: str) -> dict[str, Any] | None:
        """Получает один вектор по его ID.

        Вход:
            vector_id: идентификатор вектора.

        Выход:
            dict | None: {"id": ..., "values": [...], "metadata": {...}}
                или None, если вектор не найден.

        Исключения:
            RuntimeError: если индекс не подключён или запрос не удался.
        """
        index = self._require_index()

        try:
            response = index.fetch(ids=[vector_id])
        except PineconeError as error:
            raise RuntimeError(f"Ошибка получения вектора: {error}") from error

        vector = response.vectors.get(vector_id)
        if vector is None:
            return None

        return {
            "id": vector.id,
            "values": list(vector.values),
            "metadata": vector.metadata or {},
        }

    def search_vectors(
        self,
        query_vector: list[float],
        top_k: int = 10,
        include_metadata: bool = True,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Ищет векторы, наиболее похожие на query_vector.

        Вход:
            query_vector: вектор запроса (той же размерности, что и индекс).
            top_k: сколько ближайших результатов вернуть.
            include_metadata: возвращать ли metadata найденных векторов.
            filter_dict: фильтр по metadata, например {"genre": {"$eq": "news"}}.

        Выход:
            list[dict]: список {"id": ..., "score": ..., "metadata": {...}},
                отсортированный по убыванию похожести.

        Исключения:
            ValueError: если query_vector пуст или top_k < 1.
            RuntimeError: если индекс не подключён или запрос не удался.
        """
        index = self._require_index()

        if not query_vector:
            raise ValueError("query_vector пуст")
        if top_k < 1:
            raise ValueError("top_k должен быть не меньше 1")

        try:
            response = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=include_metadata,
                filter=filter_dict,
            )
        except PineconeError as error:
            raise RuntimeError(f"Ошибка поиска: {error}") from error

        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata or {},
            }
            for match in response.matches
        ]

    def delete(self, vector_id: str) -> None:
        """Удаляет один вектор по ID.

        Вход:
            vector_id: идентификатор вектора.

        Выход:
            None. Если вектора нет, Pinecone просто ничего не удаляет.

        Исключения:
            RuntimeError: если индекс не подключён или запрос не удался.
        """
        index = self._require_index()

        try:
            index.delete(ids=[vector_id])
        except PineconeError as error:
            raise RuntimeError(f"Ошибка удаления вектора: {error}") from error

    def delete_all(self) -> None:
        """Удаляет все векторы в индексе (сам индекс остаётся).

        Вход:
            Ничего.

        Выход:
            None.

        Исключения:
            RuntimeError: если индекс не подключён или запрос не удался.
        """
        index = self._require_index()

        try:
            index.delete(delete_all=True)
        except NotFoundError:
            # Pinecone отвечает 404, если в namespace ещё нет ни одного вектора.
            return
        except PineconeError as error:
            raise RuntimeError(f"Ошибка удаления всех векторов: {error}") from error
