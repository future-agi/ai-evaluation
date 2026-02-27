"""Run history storage for tracking evaluation runs."""

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fi.evals.types import BatchRunResult


@dataclass
class RunRecord:
    """Record of a single evaluation run."""

    run_id: str
    timestamp: str
    config_file: Optional[str]
    templates: List[str]
    total_evaluations: int
    successful: int
    failed: int
    pass_rate: Optional[float]
    avg_score: Optional[float]
    results_file: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunRecord":
        """Create a RunRecord from a dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class RunHistory:
    """Manages evaluation run history."""

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize run history storage.

        Args:
            storage_dir: Directory to store runs. Defaults to .fi/runs in cwd.
        """
        if storage_dir is None:
            storage_dir = Path.cwd() / ".fi" / "runs"

        self.storage_dir = Path(storage_dir)
        self.history_file = self.storage_dir / "history.json"
        self._ensure_storage_dir()

    def _ensure_storage_dir(self) -> None:
        """Create storage directory if it doesn't exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Create .gitignore to exclude results
        gitignore_path = self.storage_dir.parent / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text("# Ignore evaluation runs\nruns/\n")

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load run history from file."""
        if not self.history_file.exists():
            return []

        try:
            with open(self.history_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save_history(self, history: List[Dict[str, Any]]) -> None:
        """Save run history to file."""
        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)

    def save_run(
        self,
        results: BatchRunResult,
        config_file: Optional[str] = None,
        templates: Optional[List[str]] = None,
    ) -> RunRecord:
        """
        Save an evaluation run to history.

        Args:
            results: BatchRunResult from the evaluation
            config_file: Path to the config file used (if any)
            templates: List of template names used

        Returns:
            RunRecord with the saved run info
        """
        run_id = self._generate_run_id()
        timestamp = datetime.now().isoformat()

        # Calculate statistics
        total = len(results.eval_results)
        successful = len([r for r in results.eval_results if r is not None])
        failed = total - successful

        # Calculate pass rate for boolean outputs
        boolean_results = [
            r for r in results.eval_results
            if r is not None and isinstance(r.output, bool)
        ]
        passed = len([r for r in boolean_results if r.output is True])
        pass_rate = (passed / len(boolean_results) * 100) if boolean_results else None

        # Calculate average score for numeric outputs
        score_results = [
            r for r in results.eval_results
            if r is not None and isinstance(r.output, (int, float))
        ]
        avg_score = (
            sum(r.output for r in score_results) / len(score_results)
            if score_results else None
        )

        # Get unique template names
        if templates is None:
            templates = list(set(
                r.name for r in results.eval_results if r is not None
            ))

        # Save results to file
        results_file = self.storage_dir / f"{run_id}.json"
        results_data = {
            "run_id": run_id,
            "timestamp": timestamp,
            "eval_results": [
                {
                    "name": r.name,
                    "output": r.output,
                    "reason": r.reason,
                    "runtime": r.runtime,
                    "output_type": r.output_type,
                    "eval_id": r.eval_id,
                }
                for r in results.eval_results
                if r is not None
            ],
        }

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        # Create record
        record = RunRecord(
            run_id=run_id,
            timestamp=timestamp,
            config_file=config_file,
            templates=templates,
            total_evaluations=total,
            successful=successful,
            failed=failed,
            pass_rate=pass_rate,
            avg_score=avg_score,
            results_file=str(results_file),
        )

        # Add to history
        history = self._load_history()
        history.insert(0, record.to_dict())  # Most recent first

        # Keep only last 100 runs
        history = history[:100]

        self._save_history(history)

        return record

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        """Get a specific run by ID."""
        history = self._load_history()
        for entry in history:
            if entry["run_id"] == run_id:
                return RunRecord.from_dict(entry)
        return None

    def get_latest_run(self) -> Optional[RunRecord]:
        """Get the most recent run."""
        history = self._load_history()
        if history:
            return RunRecord.from_dict(history[0])
        return None

    def list_runs(self, limit: int = 10) -> List[RunRecord]:
        """List recent runs."""
        history = self._load_history()
        return [RunRecord.from_dict(entry) for entry in history[:limit]]

    def load_results(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load full results for a run."""
        record = self.get_run(run_id)
        if record is None:
            return None

        results_file = Path(record.results_file)
        if not results_file.exists():
            return None

        with open(results_file, "r") as f:
            return json.load(f)

    def delete_run(self, run_id: str) -> bool:
        """Delete a run from history."""
        history = self._load_history()

        # Find and remove the run
        new_history = [entry for entry in history if entry["run_id"] != run_id]

        if len(new_history) == len(history):
            return False  # Run not found

        # Delete results file
        for entry in history:
            if entry["run_id"] == run_id:
                results_file = Path(entry["results_file"])
                if results_file.exists():
                    results_file.unlink()
                break

        self._save_history(new_history)
        return True

    def clear_history(self) -> int:
        """Clear all run history. Returns number of runs deleted."""
        history = self._load_history()
        count = len(history)

        # Delete all results files
        for entry in history:
            results_file = Path(entry["results_file"])
            if results_file.exists():
                results_file.unlink()

        self._save_history([])
        return count

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        # Use timestamp + short UUID for readability
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"{timestamp}-{short_uuid}"
