from typing import Any

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    type: str
    description: str = Field(min_length=1)
    severity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    source_module: str
    location: dict[str, Any] | None = None
    timestamp_seconds: float | None = None