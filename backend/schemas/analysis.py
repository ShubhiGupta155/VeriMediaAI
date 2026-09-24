from enum import Enum

from pydantic import BaseModel, Field, model_validator

from backend.schemas.evidence import EvidenceItem


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


class ModuleStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class ModuleResult(BaseModel):
    analysis_id: str
    media_type: MediaType
    module: str
    status: ModuleStatus

    scores: dict[str, float] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    timestamps: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    model_versions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_result(self):
        for name, score in self.scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Score '{name}' must be between 0.0 and 1.0")

        if self.status == ModuleStatus.SUCCESS and not self.model_versions:
            raise ValueError("Successful results must include model_versions")

        if self.status == ModuleStatus.FAILED and not self.errors:
            raise ValueError("Failed results must include at least one error")

        return self