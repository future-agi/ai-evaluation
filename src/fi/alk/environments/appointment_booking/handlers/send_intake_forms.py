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
    patient = _patient(db, patient_id)
    _require_verified(db, patient_id)
    channel = str(args.get("channel") or "").strip().lower()
    if channel not in ("sms", "email"):
        raise ToolError("channel must be sms or email.")
    destination = patient["phone"] if channel == "sms" else patient["email"]
    if not destination:
        missing = "mobile number" if channel == "sms" else "email address"
        raise ToolError(f"There is no {missing} on file for this patient.")
    count = db.one("SELECT COUNT(*) AS n FROM messages")["n"]
    message_id = f"M-{7001 + count}"
    db.execute(
        "INSERT INTO messages (message_id, patient_id, kind, channel, destination) "
        "VALUES (?, ?, 'intake_forms', ?, ?)",
        (message_id, patient_id, channel, destination),
    )
    return {
        "message_id": message_id,
        "sent": True,
        "channel": channel,
        "destination": destination,
    }
