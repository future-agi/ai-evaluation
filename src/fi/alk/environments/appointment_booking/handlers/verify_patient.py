# ruff: noqa: F821 - the world runtime injects ToolError when it executes a handler.


def handle(args, db):
    full_name = " ".join(str(args.get("full_name") or "").split()).lower()
    dob = str(args.get("dob") or "").strip()
    if not full_name or not dob:
        raise ToolError("Give the patient's full name and date of birth (YYYY-MM-DD).")
    patient = db.one(
        "SELECT patient_id, full_name FROM patients WHERE lower(full_name) = ? AND dob = ?",
        (full_name, dob),
    )
    if patient is None:
        raise ToolError("No patient matches that full name and date of birth.")
    db.execute(
        "INSERT INTO verified_patients (patient_id) VALUES (?) ON CONFLICT (patient_id) DO NOTHING",
        (patient["patient_id"],),
    )
    return {
        "patient_id": patient["patient_id"],
        "full_name": patient["full_name"],
        "verified": True,
    }
