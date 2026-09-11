CREATE TABLE IF NOT EXISTS accounts (
    account_id text PRIMARY KEY,
    customer_name text NOT NULL,
    support_status text NOT NULL
);
INSERT INTO accounts(account_id, customer_name, support_status)
VALUES ('ACC-2048', 'Taylor Morgan', 'priority support active')
ON CONFLICT (account_id) DO UPDATE SET
    customer_name = EXCLUDED.customer_name,
    support_status = EXCLUDED.support_status;
