# ruff: noqa: F821 - the world runtime injects ToolError when it executes a handler.
from datetime import date


def handle(args, db):
    specialty = str(args.get("specialty") or "").strip().lower()
    known = sorted(
        row["specialty"] for row in db.query("SELECT DISTINCT specialty FROM providers")
    )
    if specialty not in known:
        raise ToolError(
            f"Unknown specialty {specialty!r}; choose one of: {', '.join(known)}."
        )
    try:
        start = date.fromisoformat(str(args.get("date_from") or "").strip())
        end = date.fromisoformat(str(args.get("date_to") or "").strip())
    except ValueError:
        raise ToolError(
            "date_from and date_to must be dates in YYYY-MM-DD form."
        ) from None
    if end < start:
        raise ToolError("date_to must not be before date_from.")
    now = db.one("SELECT now FROM clock")["now"]
    slots = db.query(
        "SELECT s.slot_id, p.name AS provider, p.specialty, s.starts_at, s.location "
        "FROM slots s JOIN providers p ON p.provider_id = s.provider_id "
        "WHERE p.specialty = ? AND substr(s.starts_at, 1, 10) BETWEEN ? AND ? "
        "AND s.starts_at > ? AND NOT EXISTS (SELECT 1 FROM appointments a "
        "WHERE a.slot_id = s.slot_id AND a.status = 'booked') ORDER BY s.starts_at",
        (specialty, start.isoformat(), end.isoformat(), now),
    )
    return {"slots": slots}
