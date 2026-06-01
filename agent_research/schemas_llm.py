from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ── Research scope ────────────────────────────────────────────────────────────

class ResearchScope(StrictModel):
    selected_symbols: list[str] | None = None
    all_symbols: bool | None = None
    start_date: str | None = None
    selected_timeframes: list[str] | None = None

    @field_validator("selected_symbols")
    @classmethod
    def validate_selected_symbols(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized = [symbol.strip() for symbol in value if str(symbol).strip()]
        if not normalized:
            raise ValueError("selected_symbols must not be an empty list when provided")
        return normalized[:15]


class StrategyFlags(StrictModel):
    requires_funding_model: bool = False
    require_positive_funding_carry: bool = False
    uses_peer_funding_signal: bool = False


class CodeChangeProposal(StrictModel):
    script_name: str
    change_goal: str
    expected_outputs: list[str] = Field(default_factory=lambda: ["signals_csv", "metrics_json"])
    expected_effect: str
    risk_level: Literal["low", "medium", "high"] = "medium"
    strategy_flags: StrategyFlags = Field(default_factory=StrategyFlags)

    @field_validator("script_name")
    @classmethod
    def validate_script_name(cls, v: str) -> str:
        name = v.strip()
        if not name.endswith(".py"):
            raise ValueError("script_name must end with .py")
        if "/" in name or "\\" in name:
            raise ValueError("script_name must be a file name, not a path")
        return name


class CodeWriterResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    filename: str
    contents: str
    entrypoint: str = "main"
    expected_output_schema: dict = Field(default_factory=dict)
    change_summary: str
    risk_level: Literal["low", "medium", "high"]

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        name = v.strip()
        if not name.endswith(".py"):
            raise ValueError("filename must end with .py")
        if "/" in name or "\\" in name:
            raise ValueError("filename must be a file name, not a path")
        return name


class CodeWriterPlanResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    filename: str
    entrypoint: str = "main"
    expected_output_schema: dict = Field(default_factory=dict)
    change_summary: str
    risk_level: Literal["low", "medium", "high"]
    total_chunks: int = Field(ge=1, le=12)

    @field_validator("filename")
    @classmethod
    def validate_plan_filename(cls, v: str) -> str:
        name = v.strip()
        if not name.endswith(".py"):
            raise ValueError("filename must end with .py")
        if "/" in name or "\\" in name:
            raise ValueError("filename must be a file name, not a path")
        return name


class CodeWriterChunkResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunk_index: int = Field(ge=1, le=12)
    total_chunks: int = Field(ge=1, le=12)
    content_chunk: str

    @field_validator("content_chunk")
    @classmethod
    def validate_content_chunk(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content_chunk must not be empty")
        return v


# ── Agent response schemas ────────────────────────────────────────────────────

class ResearcherResponse(StrictModel):
    code_direction: str
    research_scope: ResearchScope = Field(default_factory=ResearchScope)
    rationale: str
    code_change_proposal: CodeChangeProposal | None = None


class HypothesisProposal(StrictModel):
    title: str
    claim: str
    rationale: str
    research_questions: list[str]
    success_criteria: str


class StrategistResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Synthesis
    research_memo: str = ""
    patterns_detected: list[str] = Field(default_factory=list)
    new_hypotheses: list[HypothesisProposal] = Field(default_factory=list)
    hypotheses_to_close: list[str] = Field(default_factory=list)

    # Planning
    active_hypothesis_id: str = "H1"
    code_direction: str = ""
    research_scope: ResearchScope = Field(default_factory=ResearchScope)
    rationale: str = ""
    suggested_agenda: list[str] = Field(default_factory=list)


class ReviewerResponse(StrictModel):
    verdict: Literal["approve", "reject"]
    notes: str
    root_cause: str | None = None
    failed_dimensions: list[str] | None = None
    passing_dimensions: list[str] | None = None
    diagnostic_insight: str | None = None
    suggested_direction: str | None = None
    cross_strategy_note: str | None = None


class RouterResponse(StrictModel):
    decision: Literal["next_iteration", "switch_hypothesis", "done"]
    message: str = ""
    next_hypothesis_id: str = ""


RESPONSE_MODELS = {
    "strategist":  StrategistResponse,
    "researcher":  ResearcherResponse,
    "reviewer":    ReviewerResponse,
    "router":      RouterResponse,
    "coder":       CodeWriterResponse,
    "coder_plan":  CodeWriterPlanResponse,
    "coder_chunk": CodeWriterChunkResponse,
}


def validate_llm_response(agent_name: str, payload: dict, schema_name: str | None = None) -> dict:
    model_cls = RESPONSE_MODELS.get((schema_name or agent_name or "").lower())
    if model_cls is None:
        return payload
    model = model_cls.model_validate(payload)
    return model.model_dump(exclude_none=True)


def format_validation_error(error: ValidationError) -> str:
    parts: list[str] = []
    for issue in error.errors():
        location = ".".join(str(item) for item in issue.get("loc", ()))
        message = issue.get("msg", "invalid value")
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts)
