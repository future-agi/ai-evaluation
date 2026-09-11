from __future__ import annotations

from fi.alk import trinity
from fi.alk.cli import main


def test_doctor_returns_nonzero_when_boundary_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        trinity,
        "trinity_status",
        lambda: {
            "status": "failed",
            "exit_code": 1,
            "summary": {
                "missing_public_modules": ["fi.alk.simulate"],
                "missing_engine_modules": [],
            },
        },
    )

    assert main(["doctor", "--quiet"]) == 1
