"""Tool and event backend for importing an existing Vapi or Retell target."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request


app = FastAPI()
_trace = Path(os.getenv("PROVIDER_TRACE_PATH", "/tmp/provider-trace.jsonl"))

_patients = [
    {
        "patient_id": "pat_001",
        "full_name": "Jane Doe",
        "date_of_birth": "1985-04-12",
        "phone": "+15551234567",
    },
    {
        "patient_id": "pat_002",
        "full_name": "John Smith",
        "date_of_birth": "1972-09-28",
        "phone": "+15559876543",
    },
    {
        "patient_id": "pat_003",
        "full_name": "Alice Taylor",
        "date_of_birth": "1990-11-05",
        "phone": "+15554567890",
    },
]
_prescriptions = [
    {
        "rx_id": "rx_101",
        "patient_id": "pat_001",
        "medication_name": "Amoxicillin 500mg",
        "status": "active",
        "refills_remaining": 2,
        "is_controlled": False,
    },
    {
        "rx_id": "rx_102",
        "patient_id": "pat_001",
        "medication_name": "Lisinopril 10mg",
        "status": "no_refills_remaining",
        "refills_remaining": 0,
        "is_controlled": False,
    },
    {
        "rx_id": "rx_103",
        "patient_id": "pat_002",
        "medication_name": "Zolpidem 5mg",
        "status": "controlled_substance",
        "refills_remaining": 1,
        "is_controlled": True,
    },
    {
        "rx_id": "rx_104",
        "patient_id": "pat_002",
        "medication_name": "Atorvastatin 20mg",
        "status": "expired",
        "refills_remaining": 3,
        "is_controlled": False,
    },
    {
        "rx_id": "rx_105",
        "patient_id": "pat_003",
        "medication_name": "Omeprazole 20mg",
        "status": "too_soon",
        "refills_remaining": 1,
        "is_controlled": False,
    },
]
_refills = {
    "ref_201": {
        "refill_id": "ref_201",
        "rx_id": "rx_101",
        "status": "ready_for_collection",
    },
    "ref_202": {"refill_id": "ref_202", "rx_id": "rx_105", "status": "processing"},
}


def _record(kind: str, body: Any) -> None:
    _trace.parent.mkdir(parents=True, exist_ok=True)
    with _trace.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "kind": kind,
                    "body": body,
                },
                sort_keys=True,
                default=str,
            )
            + "\n"
        )


def _arguments(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {}
    arguments = body.get("args") or body.get("arguments") or body
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return {}
    return arguments if isinstance(arguments, dict) else {}


async def _tool_request(request: Request, name: str) -> dict[str, Any]:
    body = await request.json()
    _record(f"tool.{name}", body)
    return _arguments(body)


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/provider/events")
async def provider_events(request: Request) -> dict[str, bool]:
    _record("provider_event", await request.json())
    return {"received": True}


@app.post("/provider/tools/record_preference")
async def record_preference(request: Request) -> dict[str, Any]:
    body = await request.json()
    _record("tool.record_preference", body)
    preference = _arguments(body).get("preference")
    return {
        "recorded": True,
        "preference": preference,
    }


@app.post("/provider/tools/find_patient")
async def find_patient(request: Request) -> dict[str, Any]:
    arguments = await _tool_request(request, "find_patient")
    phone = str(arguments.get("phone") or "").strip()
    full_name = str(arguments.get("full_name") or "").strip().casefold()
    date_of_birth = str(arguments.get("date_of_birth") or "").strip()
    for patient in _patients:
        phone_match = phone and patient["phone"] == phone
        identity_match = (
            full_name
            and date_of_birth
            and patient["full_name"].casefold() == full_name
            and patient["date_of_birth"] == date_of_birth
        )
        if phone_match or identity_match:
            return {"found": True, "patient": patient}
    return {"found": False, "patient": None}


@app.post("/provider/tools/list_prescriptions")
async def list_prescriptions(request: Request) -> dict[str, Any]:
    arguments = await _tool_request(request, "list_prescriptions")
    patient_id = str(arguments.get("patient_id") or "").strip()
    return {
        "patient_id": patient_id,
        "prescriptions": [
            prescription
            for prescription in _prescriptions
            if prescription["patient_id"] == patient_id
        ],
    }


@app.post("/provider/tools/request_refill")
async def request_refill(request: Request) -> dict[str, Any]:
    arguments = await _tool_request(request, "request_refill")
    patient_id = str(arguments.get("patient_id") or "").strip()
    rx_id = str(arguments.get("rx_id") or "").strip()
    prescription = next(
        (
            item
            for item in _prescriptions
            if item["patient_id"] == patient_id and item["rx_id"] == rx_id
        ),
        None,
    )
    if prescription is None:
        return {"accepted": False, "reason": "prescription_not_found"}
    refusal = {
        "too_soon": "too_soon",
        "no_refills_remaining": "no_refills_remaining",
        "expired": "prescription_expired",
        "controlled_substance": "controlled_medication_requires_prescriber",
    }.get(str(prescription["status"]))
    if refusal:
        return {"accepted": False, "reason": refusal, "rx_id": rx_id}
    refill_id = f"ref_{rx_id.removeprefix('rx_')}"
    _refills[refill_id] = {
        "refill_id": refill_id,
        "rx_id": rx_id,
        "status": "processing",
    }
    return {"accepted": True, **_refills[refill_id]}


@app.post("/provider/tools/check_refill_status")
async def check_refill_status(request: Request) -> dict[str, Any]:
    arguments = await _tool_request(request, "check_refill_status")
    refill_id = str(arguments.get("refill_id") or "").strip()
    refill = _refills.get(refill_id)
    return {"found": refill is not None, "refill": refill}


@app.post("/provider/tools/transfer_to_pharmacist")
async def transfer_to_pharmacist(request: Request) -> dict[str, Any]:
    arguments = await _tool_request(request, "transfer_to_pharmacist")
    return {
        "transferred": True,
        "reason": str(arguments.get("reason") or "clinical_question"),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
