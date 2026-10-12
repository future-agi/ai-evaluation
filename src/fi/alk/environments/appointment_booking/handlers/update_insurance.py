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
    payer = str(args.get("payer") or "").strip()
    member_id = str(args.get("member_id") or "").strip()
    if not payer or not member_id:
        raise ToolError("Give the insurance payer and member ID.")
    group_number = str(args.get("group_number") or "").strip()
    db.execute(
        "INSERT INTO insurance (patient_id, payer, member_id, group_number) VALUES (?, ?, ?, ?) "
        "ON CONFLICT (patient_id) DO UPDATE SET payer = excluded.payer, "
        "member_id = excluded.member_id, group_number = excluded.group_number",
        (patient_id, payer, member_id, group_number),
    )
    return {
        "patient_id": patient_id,
        "payer": payer,
        "member_id": member_id,
        "group_number": group_number,
    }
