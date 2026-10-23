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


def handle(args, db):
    patient_id = str(args.get("patient_id") or "").strip()
    _patient(db, patient_id)
    _require_verified(db, patient_id)
    appointments = db.query(
        "SELECT a.appointment_id, a.status, a.reason, s.slot_id, p.name AS provider, "
        "s.starts_at, s.location FROM appointments a JOIN slots s ON s.slot_id = a.slot_id "
        "JOIN providers p ON p.provider_id = s.provider_id WHERE a.patient_id = ? "
        "ORDER BY s.starts_at",
        (patient_id,),
    )
    return {"appointments": appointments}
