from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

from pydantic import BaseModel, Field, JsonValue, model_validator

from fi.simulate.artifacts import ArtifactManifest
from fi.simulate.evidence import EvidenceSourceSummary
from fi.simulate.simulation.models import Persona, TestCaseResult, TestReport

from .failures import SimulationFailure
from fi.simulate._hashing import content_hash
from .ids import derive_test_case_id
from .run import CleanupStatus, RunStatus, TestCaseStatus

SIMULATION_REPORT_SCHEMA_VERSION = "futureagi.simulation-report.v1"


class SimulationTestCaseResult(BaseModel):
    test_case_id: str
    status: TestCaseStatus
    persona: Persona
    result: TestCaseResult | None = None
    failure: SimulationFailure | None = None
    evidence: list[EvidenceSourceSummary] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @classmethod
    def from_legacy_case(
        cls,
        result: TestCaseResult,
        *,
        index: int,
        run_id: str,
        evidence: list[EvidenceSourceSummary] | None = None,
    ) -> "SimulationTestCaseResult":
        """Convert one legacy ``TestCaseResult`` into its canonical form.

        Shared by ``SimulationReport.from_legacy`` (report-level) and the hosted
        runner's per-case streaming callback, so a case streamed mid-run is
        byte-identical to the one the finalized report carries.
        """
        persona_ref = result.persona.version or result.persona.content_hash()
        test_case_id = str(
            result.metadata.get("test_case_id")
            or derive_test_case_id(run_id, persona_ref, index)
        )
        case_status = TestCaseStatus(
            result.metadata.get("status", TestCaseStatus.COMPLETED.value)
        )
        raw_failure = result.metadata.get("failure")
        failure = (
            SimulationFailure.model_validate(raw_failure)
            if isinstance(raw_failure, Mapping)
            else None
        )
        return cls(
            test_case_id=test_case_id,
            status=case_status,
            persona=result.persona,
            result=result,
            failure=failure,
            evidence=[item.model_copy(deep=True) for item in evidence or []],
        )

    @model_validator(mode="after")
    def _validate_outcome(self) -> "SimulationTestCaseResult":
        if self.status == TestCaseStatus.COMPLETED and self.result is None:
            raise ValueError("test_case_result_missing: completed case requires result")
        failure_statuses = {
            TestCaseStatus.FAILED,
            TestCaseStatus.TIMED_OUT,
            TestCaseStatus.AGENT_UNAVAILABLE,
        }
        if self.status in failure_statuses and self.failure is None:
            raise ValueError("test_case_failure_missing: failed case requires failure")
        return self


class SimulationReport(BaseModel):
    schema_version: str = SIMULATION_REPORT_SCHEMA_VERSION
    run_id: str
    plan_id: str | None = None
    spec_hash: str
    status: RunStatus
    cleanup_status: CleanupStatus = CleanupStatus.PENDING
    started_at: datetime
    ended_at: datetime | None = None
    test_cases: list[SimulationTestCaseResult] = Field(default_factory=list)
    artifacts: ArtifactManifest
    failure: SimulationFailure | None = None
    cleanup_failure: SimulationFailure | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    report_hash: str | None = None

    def content_hash(self) -> str:
        payload = self.model_dump(exclude={"report_hash"}, exclude_none=True)
        payload["test_cases"] = sorted(
            payload["test_cases"], key=lambda item: item["test_case_id"]
        )
        return content_hash(payload)

    def to_legacy(self, *, include_runtime_metadata: bool = True) -> TestReport:
        results = []
        for case in self.test_cases:
            if case.result is not None:
                result = case.result.model_copy(deep=True)
            else:
                result = TestCaseResult(persona=case.persona, transcript="")
            if include_runtime_metadata:
                result.metadata.update(
                    {
                        "run_id": self.run_id,
                        "test_case_id": case.test_case_id,
                        "status": case.status.value,
                    }
                )
            if case.failure is not None:
                result.metadata["failure"] = case.failure.model_dump(
                    mode="json", exclude_none=True
                )
            results.append(result)
        return TestReport(results=results)

    @classmethod
    def from_legacy(
        cls,
        report: TestReport,
        *,
        run_id: str,
        spec_hash: str,
        status: RunStatus = RunStatus.COMPLETED,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        plan_id: str | None = None,
        artifacts: ArtifactManifest | None = None,
        evidence: list[EvidenceSourceSummary] | None = None,
    ) -> "SimulationReport":
        cases = [
            SimulationTestCaseResult.from_legacy_case(
                result, index=index, run_id=run_id, evidence=evidence
            )
            for index, result in enumerate(report.results)
        ]
        return cls(
            run_id=run_id,
            plan_id=plan_id,
            spec_hash=spec_hash,
            status=status,
            started_at=started_at or datetime.now(timezone.utc),
            ended_at=ended_at,
            test_cases=cases,
            artifacts=artifacts or ArtifactManifest(run_id=run_id),
        )

    @model_validator(mode="after")
    def _validate_and_stamp(self) -> "SimulationReport":
        if self.schema_version != SIMULATION_REPORT_SCHEMA_VERSION:
            raise ValueError(
                f"simulation_report_version_unsupported: {self.schema_version}"
            )
        if self.artifacts.run_id != self.run_id:
            raise ValueError("simulation_report_artifact_run_mismatch")
        if self.status == RunStatus.FAILED and self.failure is None:
            raise ValueError("simulation_report_failure_missing")
        expected = self.content_hash()
        if self.report_hash is not None and self.report_hash != expected:
            raise ValueError("simulation_report_hash_mismatch")
        object.__setattr__(self, "report_hash", expected)
        return self
