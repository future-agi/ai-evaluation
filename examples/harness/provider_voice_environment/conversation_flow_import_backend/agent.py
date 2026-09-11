"""Deterministic certification backend for Retell appointment Conversation Flows.

This fixture proves that imported custom tools are routed into an isolated ALK
environment. It is intentionally test data, not a synthesized implementation for
customer agents. Production imports must submit their actual backend source.
"""

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
_appointments: dict[str, dict[str, Any]] = {
    "BK-20482": {
        "booking_id": "BK-20482",
        "rider_name": "Alex Morgan",
        "date_of_birth": "1985/06/15",
        "appointment_date": "September 10, 2026",
        "appointment_time": "9:00 AM",
        "pickup_location": "101 Market Street",
        "dropoff_location": "City Medical Center",
        "driver_name": "Jordan Lee",
        "booking_status": "confirmed",
        "vehicle_type": "wheelchair accessible van",
        "insurance_auth_number": "AUTH-7781",
    }
}


async def _arguments(request: Request) -> dict[str, Any]:
    body = await request.json()
    if not isinstance(body, dict):
        return {}
    nested = body.get("args") or body.get("arguments")
    if isinstance(nested, str):
        try:
            nested = json.loads(nested)
        except ValueError:
            nested = None
    return nested if isinstance(nested, dict) else body


def _record(name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
    _trace.parent.mkdir(parents=True, exist_ok=True)
    with _trace.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "kind": f"tool.{name}",
                    "arguments": arguments,
                    "result": result,
                },
                sort_keys=True,
            )
            + "\n"
        )


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/provider/events")
async def provider_events(request: Request) -> dict[str, bool]:
    body = await request.json()
    _record(
        "provider_event", body if isinstance(body, dict) else {}, {"received": True}
    )
    return {"received": True}


@app.post("/provider/tools/api/fetch-rider-appointment")
async def fetch_appointment_details(request: Request) -> dict[str, Any]:
    arguments = await _arguments(request)
    rider_name = str(arguments.get("rider_name") or "").casefold()
    date_of_birth = str(arguments.get("date_of_birth") or "")
    appointment = next(
        (
            value
            for value in _appointments.values()
            if value["rider_name"].casefold() == rider_name
            and value["date_of_birth"] == date_of_birth
        ),
        None,
    )
    result = {"booking_found": appointment is not None, **(appointment or {})}
    _record("fetch_appointment_details", arguments, result)
    return result


@app.post("/provider/tools/api/update-rider-appointment")
async def update_appointment(request: Request) -> dict[str, Any]:
    arguments = await _arguments(request)
    booking_id = str(arguments.get("booking_id") or "")
    appointment = _appointments.get(booking_id)
    updates = {
        key.removeprefix("new_"): value
        for key, value in arguments.items()
        if key.startswith("new_") and value not in (None, "")
    }
    if appointment is not None:
        appointment.update(updates)
    result = {
        "update_success": appointment is not None,
        "updated_fields": sorted(updates) if appointment is not None else [],
        "confirmation_message": (
            f"Booking {booking_id} was updated."
            if appointment is not None
            else "Booking was not found."
        ),
    }
    _record("update_appointment", arguments, result)
    return result


@app.post("/provider/tools/api/cancel-rider-appointment")
async def cancel_appointment(request: Request) -> dict[str, Any]:
    arguments = await _arguments(request)
    booking_id = str(arguments.get("booking_id") or "")
    appointment = _appointments.get(booking_id)
    if appointment is not None:
        appointment["booking_status"] = "canceled"
    result = {
        "cancellation_success": appointment is not None,
        "cancellation_confirmation": (
            f"Booking {booking_id} was canceled."
            if appointment is not None
            else "Booking was not found."
        ),
        "late_fee_applied": False,
    }
    _record("cancel_appointment", arguments, result)
    return result


@app.post("/provider/tools/api/create-rider-booking")
async def create_booking(request: Request) -> dict[str, Any]:
    arguments = await _arguments(request)
    booking_id = f"BK-{30000 + len(_appointments)}"
    appointment = {
        "booking_id": booking_id,
        "booking_status": "confirmed",
        "driver_name": "Taylor Kim",
        **arguments,
    }
    _appointments[booking_id] = appointment
    result = {
        "booking_success": True,
        "new_booking_id": booking_id,
        "confirmation_number": f"CONF-{booking_id.removeprefix('BK-')}",
        "new_booking_status": "confirmed",
        "new_driver_name": appointment["driver_name"],
        "new_pickup_location": appointment.get("pickup_location"),
        "new_dropoff_location": appointment.get("dropoff_location"),
        "new_appointment_date": appointment.get("appointment_date"),
        "new_appointment_time": appointment.get("appointment_time"),
        "new_vehicle_type": appointment.get("vehicle_type"),
        "estimated_cost": "covered by plan",
    }
    _record("create_booking", arguments, result)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
