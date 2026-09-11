def handle(db, account_id):
    rows = db.query(
        "SELECT account_id, customer_name, support_status FROM accounts WHERE account_id = ?",
        (account_id,),
    )
    return rows[0] if rows else {"error": "account_not_found"}
