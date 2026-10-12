# ruff: noqa: F821 - the world runtime injects ToolError when it executes a handler.


def _patient(db, patient_id):
    patient = db.one("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    if patient is None:
        raise ToolError(f"No patient {patient_id!r}.")
    return patient


def handle(args, db):
    team = str(args.get("team") or "").strip()
    if team not in ("front_desk", "nurse"):
        raise ToolError("team must be front_desk or nurse.")
    reason = str(args.get("reason") or "").strip()
    if not reason:
        raise ToolError("Give the reason for the handoff.")
    patient_id = str(args.get("patient_id") or "").strip() or None
    if patient_id is not None:
        _patient(db, patient_id)
    count = db.one("SELECT COUNT(*) AS n FROM escalations")["n"]
    escalation_id = f"E-{8001 + count}"
    db.execute(
        "INSERT INTO escalations (escalation_id, team, reason, patient_id) VALUES (?, ?, ?, ?)",
        (escalation_id, team, reason, patient_id),
    )
    return {"escalation_id": escalation_id, "team": team}
