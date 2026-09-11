from __future__ import annotations

import json
import os
import re
from pathlib import Path

from fi.simulate.runtime import CanonicalEvent, SimulationPlan, SimulationReport, SimulationSpec

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class LocalFilesystemResultSink:
    def __init__(self, root: str | Path = ".fagi/runs") -> None:
        self.root = Path(root).expanduser().resolve()
        self.run_directory: Path | None = None

    def prepare(
        self,
        spec: SimulationSpec,
        plan: SimulationPlan | None = None,
    ) -> Path:
        if not _SAFE_RUN_ID.fullmatch(spec.run_id) or ".." in spec.run_id:
            raise ValueError("run_id_invalid: run_id is not filesystem safe")
        run_directory = self.root / spec.run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        (run_directory / "audio").mkdir(exist_ok=True)
        (run_directory / "logs").mkdir(exist_ok=True)
        self.run_directory = run_directory
        self._write_json("spec.json", spec.model_dump(mode="json", exclude_none=True))
        if plan is not None:
            self._write_json(
                "plan.json",
                plan.model_dump(mode="json", exclude_none=True),
            )
        return run_directory

    def write_event(self, event: CanonicalEvent) -> None:
        run_directory = self._require_prepared()
        with (run_directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(event.model_dump_json(exclude_none=True))
            stream.write("\n")

    def write_report(self, report: SimulationReport) -> Path:
        if report.run_id != self._require_prepared().name:
            raise ValueError("result_sink_run_mismatch")
        self._write_json(
            "artifacts.json",
            report.artifacts.model_dump(mode="json", exclude_none=True),
        )
        return self._write_json(
            "report.json",
            report.model_dump(mode="json", exclude_none=True),
        )

    def _write_json(self, name: str, payload: object) -> Path:
        run_directory = self._require_prepared()
        destination = run_directory / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        return destination

    def _require_prepared(self) -> Path:
        if self.run_directory is None:
            raise RuntimeError("result_sink_not_prepared")
        return self.run_directory
