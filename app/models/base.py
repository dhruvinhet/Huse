"""Shared Pydantic model configuration and reusable field types."""

from typing import Annotated

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, StringConstraints


NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class BaseModel(PydanticBaseModel):
    """Base class that applies strict configuration to pipeline models."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=False,
    )
