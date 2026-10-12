# ruff: noqa: F821 - the world runtime injects ToolError when it executes a handler.
import json


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


def _patient(db, patient_id):
    patient = db.one("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    if patient is None:
        raise ToolError(f"No patient {patient_id!r}.")
    return patient


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
    patient_id = str(args.get("patient_id") or "").strip()
    _patient(db, patient_id)
    _require_verified(db, patient_id)
    reason = str(args.get("reason") or "").strip()
    if not reason:
        raise ToolError("Give a reason for the visit.")
    slot = _open_slot(db, str(args.get("slot_id") or "").strip())
    count = db.one("SELECT COUNT(*) AS n FROM appointments")["n"]
    appointment_id = f"A-{6001 + count}"
    db.execute(
        "INSERT INTO appointments (appointment_id, patient_id, slot_id, reason, status) "
        "VALUES (?, ?, ?, ?, 'booked')",
        (appointment_id, patient_id, slot["slot_id"], reason),
    )
    return {"appointment_id": appointment_id, **slot}
