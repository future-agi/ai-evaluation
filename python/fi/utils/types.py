from enum import Enum, unique
from typing import List, NamedTuple, Optional, Sequence, Type, TypeVar, Union

import numpy as np
import pandas as pd

from fi.utils.constants import (
    MAX_RAW_DATA_CHARACTERS,
    MAX_RAW_DATA_CHARACTERS_TRUNCATION,
)
from fi.utils.logging import get_truncation_warning_message, logger


class ModelTypes(Enum):
    """Model types for dataset"""

    GENERATIVE_LLM = "GenerativeLLM"
    GENERATIVE_IMAGE = "GenerativeImage"

    @classmethod
    def get_choices(cls):
        return [(tag.value, tag.name.replace("_", " ").title()) for tag in cls]


@unique
class Environments(Enum):
    TRAINING = 1
    VALIDATION = 2
    PRODUCTION = 3
    CORPUS = 4


class Embedding(NamedTuple):
    vector: List[float]
    data: Optional[Union[str, List[str]]] = None
    link_to_data: Optional[str] = None

    def validate(self, emb_name: Union[str, int, float]) -> None:
        """
        Validates that the embedding object passed is of the correct format. That is, validations must
        be passed for vector, data & link_to_data.

        Arguments:
        ----------
            emb_name (str, int, float): Name of the embedding feature the vector belongs to

        Raises:
        -------
            TypeError: If the embedding fields are of the wrong type
        """

        if self.vector is not None:
            self._validate_embedding_vector(emb_name)

        # Validate embedding raw data, if present
        if self.data is not None:
            self._validate_embedding_data(emb_name, self.data)

        # Validate embedding link to data, if present
        if self.link_to_data is not None:
            self._validate_embedding_link_to_data(emb_name, self.link_to_data)

        return None

    def _validate_embedding_vector(
        self,
        emb_name: Union[str, int, float],
    ) -> None:
        """
        Validates that the embedding vector passed is of the correct format. That is:
            1. Type must be list or convertible to list (like numpy arrays, pandas Series)
            2. List must not be empty
            3. Elements in list must be floats

        Arguments:
        ----------
            emb_name (str, int, float): Name of the embedding feature the vector belongs to

        Raises:
        -------
            TypeError: If the embedding does not satisfy requirements above
        """

        if not Embedding._is_valid_iterable(self.vector):
            raise TypeError(
                f'Embedding feature "{emb_name}" has vector type {type(self.vector)}. Must be '
                f"list, "
                f"np.ndarray or pd.Series"
            )
        # Fail if not all elements in list are floats
        allowed_types = (int, float, np.int16, np.int32, np.float16, np.float32)
        if not all(isinstance(val, allowed_types) for val in self.vector):  # type: ignore
            raise TypeError(
                f"Embedding vector must be a vector of integers and/or floats. Got "
                f"{emb_name}.vector = {self.vector}"
            )
        # Fail if the length of the vector is 1
        if len(self.vector) == 1:
            raise ValueError("Embedding vector must not have a size of 1")

    @staticmethod
    def _validate_embedding_data(
        emb_name: Union[str, int, float], data: Union[str, List[str]]
    ) -> None:
        """
        Validates that the embedding raw data field is of the correct format. That is:
            1. Must be string or list of strings (NLP case)

        Arguments:
        ----------
            emb_name (str, int, float): Name of the embedding feature the vector belongs to
            data (str, int, float): Raw data associated with the embedding feature. Typically raw text.

        Raises:
        -------
            TypeError: If the embedding does not satisfy requirements above
        """
        # Validate that data is a string or iterable of strings
        is_string = isinstance(data, str)
        is_allowed_iterable = not is_string and Embedding._is_valid_iterable(data)
        if not (is_string or is_allowed_iterable):
            raise TypeError(
                f'Embedding feature "{emb_name}" data field must be str, list, np.ndarray or '
                f"pd.Series"
            )

        if is_allowed_iterable:
            # Fail if not all elements in iterable are strings
            if not all(isinstance(val, str) for val in data):
                raise TypeError("Embedding data field must contain strings")

        character_count = count_characters_raw_data(data)
        if character_count > MAX_RAW_DATA_CHARACTERS:
            raise ValueError(
                f"Embedding data field must not contain more than {MAX_RAW_DATA_CHARACTERS} characters. "
                f"Found {character_count}."
            )
        elif character_count > MAX_RAW_DATA_CHARACTERS_TRUNCATION:
            logger.warning(
                get_truncation_warning_message(
                    "Embedding raw data fields", MAX_RAW_DATA_CHARACTERS_TRUNCATION
                )
            )

    @staticmethod
    def _validate_embedding_link_to_data(
        emb_name: Union[str, int, float], link_to_data: str
    ) -> None:
        """
        Validates that the embedding link to data field is of the correct format. That is:
            1. Must be string

        Arguments:
        ----------
            emb_name (str, int, float): Name of the embedding feature the vector belongs to
            link_to_data (str): Link to source data of embedding feature, typically an image file on
                cloud storage

        Raises:
        -------
            TypeError: If the embedding does not satisfy requirements above
        """
        if not isinstance(link_to_data, str):
            raise TypeError(
                f'Embedding feature "{emb_name}" link_to_data field must be str and got '
                f"{type(link_to_data)}"
            )

    @staticmethod
    def _is_valid_iterable(
        data: Union[str, List[str], List[float], np.ndarray, pd.Series]
    ) -> bool:
        """
        Validates that the input data field is of the correct iterable type. That is:
            1. List or
            2. numpy array or
            3. pandas Series

        Arguments:
        ----------
            data: input iterable

        Returns:
        --------
            True if the data type is one of the accepted iterable types, false otherwise
        """
        return any(isinstance(data, t) for t in (list, np.ndarray, pd.Series))


T = TypeVar("T")


def is_list_of(lst: Sequence[object], tp: Type[T]) -> bool:
    return isinstance(lst, list) and all(isinstance(x, tp) for x in lst)


def count_characters_raw_data(data: Union[str, List[str]]) -> int:
    character_count = 0
    if isinstance(data, str):
        return len(data)
    for string in data:
        character_count += len(string)
    return character_count
