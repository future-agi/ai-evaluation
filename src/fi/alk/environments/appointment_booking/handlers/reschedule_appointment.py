# ruff: noqa: F821 - the world runtime injects ToolError when it executes a handler.
import json
from datetime import datetime, timedelta


def _setting(db, name):
    row = db.one("SELECT value FROM environment_policy WHERE name = ?", (name,))
    return json.loads(row["value"]) if row else None


def _require_verified(db, patient_id):
    if _setting(db, "verification_required") and not db.one(
        "SELECT patient_id FROM verified_patients WHERE patient_id = ?", (patient_id,)
    ):
        raise ToolError(
            "Verify the patient with verify_patient before accessing or changing their records."
        )


def _inside_notice(db, starts_at):
    hours = _setting(db, "change_notice_hours") or 0
    now = datetime.fromisoformat(db.one("SELECT now FROM clock")["now"])
    return datetime.fromisoformat(starts_at) - now < timedelta(hours=hours), hours


def _open_slot(db, slot_id):
    slot = db.one(
        "SELECT s.slot_id, p.name AS provider, s.starts_at, s.location FROM slots s "
        "JOIN providers p ON p.provider_id = s.provider_id WHERE s.slot_id = ?",
        (slot_id,),
    )
    if slot is None:
        raise ToolError(f"No slot {slot_id!r}.")
    if slot["starts_at"] <= db.one("SELECT now FROM clock")["now"]:
        raise ToolError("That slot has already started.")
    if db.one(
        "SELECT appointment_id FROM appointments WHERE slot_id = ? AND status = 'booked'",
        (slot_id,),
    ):
        raise ToolError("That slot is already booked.")
    return slot


def handle(args, db):
    appointment_id = str(args.get("appointment_id") or "").strip()
    appointment = db.one(
        "SELECT a.appointment_id, a.patient_id, a.status, a.slot_id, s.starts_at "
        "FROM appointments a JOIN slots s ON s.slot_id = a.slot_id WHERE a.appointment_id = ?",
        (appointment_id,),
    )
    if appointment is None:
        raise ToolError(f"No appointment {appointment_id!r}.")
    _require_verified(db, appointment["patient_id"])
    if appointment["status"] != "booked":
        raise ToolError(
            f"Appointment {appointment_id} is cancelled and cannot be moved; book a new one instead."
        )
    inside, hours = _inside_notice(db, appointment["starts_at"])
    if inside:
        raise ToolError(
            f"Appointments cannot be changed less than {hours} hours before they start."
        )
    new_slot_id = str(args.get("new_slot_id") or "").strip()
    if new_slot_id == appointment["slot_id"]:
        raise ToolError("The appointment is already in that slot.")
    slot = _open_slot(db, new_slot_id)
    db.execute(
        "UPDATE appointments SET slot_id = ? WHERE appointment_id = ?",
        (slot["slot_id"], appointment_id),
    )
    return {"appointment_id": appointment_id, **slot}
