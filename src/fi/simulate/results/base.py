from __future__ import annotations

from pathlib import Path
from typing import Protocol

from fi.simulate.runtime import CanonicalEvent, SimulationPlan, SimulationReport, SimulationSpec


class ResultSink(Protocol):
    def prepare(
        self,
        spec: SimulationSpec,
        plan: SimulationPlan | None = None,
    ) -> Path: ...

    def write_event(self, event: CanonicalEvent) -> None: ...

    def write_report(self, report: SimulationReport) -> Path: ...
