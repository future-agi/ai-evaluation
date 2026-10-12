-- The clinic's world: schema, invariants and seed. Times are clinic-local ISO strings and the
-- clock is fixed, so every scenario starts from the same "now".

CREATE TABLE clock (now TEXT NOT NULL);

CREATE TABLE environment_policy (name TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE patients (
    patient_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    dob TEXT NOT NULL,
    phone TEXT,
    email TEXT,
    is_new_patient INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE providers (
    provider_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    specialty TEXT NOT NULL
);

CREATE TABLE slots (
    slot_id TEXT PRIMARY KEY,
    provider_id TEXT NOT NULL REFERENCES providers (provider_id),
    starts_at TEXT NOT NULL,
    location TEXT NOT NULL
);

CREATE TABLE appointments (
    appointment_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients (patient_id),
    slot_id TEXT NOT NULL REFERENCES slots (slot_id),
    reason TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('booked', 'cancelled')),
    cancel_reason TEXT
);

-- no_double_booking, enforced by the store as well as by the handlers.
CREATE UNIQUE INDEX one_booking_per_slot ON appointments (slot_id) WHERE status = 'booked';

CREATE TABLE insurance (
    patient_id TEXT PRIMARY KEY REFERENCES patients (patient_id),
    payer TEXT NOT NULL,
    member_id TEXT NOT NULL,
    group_number TEXT NOT NULL DEFAULT ''
);

CREATE TABLE verified_patients (
    patient_id TEXT PRIMARY KEY REFERENCES patients (patient_id)
);

CREATE TABLE messages (
    message_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients (patient_id),
    kind TEXT NOT NULL,
    channel TEXT NOT NULL CHECK (channel IN ('sms', 'email')),
    destination TEXT NOT NULL
);

CREATE TABLE escalations (
    escalation_id TEXT PRIMARY KEY,
    team TEXT NOT NULL CHECK (team IN ('front_desk', 'nurse')),
    reason TEXT NOT NULL,
    patient_id TEXT REFERENCES patients (patient_id)
);

INSERT INTO clock (now) VALUES ('2026-03-02T09:00:00');

INSERT INTO environment_policy (name, value) VALUES
    ('verification_required', 'true'),
    ('change_notice_hours', '24');

INSERT INTO patients (patient_id, full_name, dob, phone, email, is_new_patient) VALUES
    ('P-1001', 'Marcus Vance', '1982-04-12', '+15550101001', 'marcus.vance@example.com', 0),
    ('P-1002', 'Priya Raman', '1990-11-03', '+15550101002', NULL, 0),
    ('P-1003', 'Elena Ortiz', '1975-06-21', NULL, 'elena.ortiz@example.com', 0),
    ('P-1004', 'Daniel Kim', '2001-02-14', '+15550101004', 'daniel.kim@example.com', 1),
    ('P-1005', 'Grace Okafor', '1968-09-30', '+15550101005', 'grace.okafor@example.com', 0);

INSERT INTO providers (provider_id, name, specialty) VALUES
    ('DR-NGUYEN', 'Dr. Alicia Nguyen', 'primary_care'),
    ('DR-PATEL', 'Dr. Rohan Patel', 'primary_care'),
    ('DR-LEE', 'Dr. Hannah Lee', 'pediatrics'),
    ('DR-MORALES', 'Dr. Sofia Morales', 'dermatology');

INSERT INTO slots (slot_id, provider_id, starts_at, location) VALUES
    ('S-100', 'DR-NGUYEN', '2026-03-02T15:00:00', 'Main Clinic'),
    ('S-101', 'DR-NGUYEN', '2026-03-03T10:00:00', 'Main Clinic'),
    ('S-102', 'DR-NGUYEN', '2026-03-04T14:30:00', 'Main Clinic'),
    ('S-103', 'DR-NGUYEN', '2026-03-06T09:00:00', 'Main Clinic'),
    ('S-104', 'DR-NGUYEN', '2026-03-09T11:00:00', 'Main Clinic'),
    ('S-105', 'DR-NGUYEN', '2026-03-11T15:30:00', 'Main Clinic'),
    ('S-106', 'DR-PATEL', '2026-03-03T13:00:00', 'Eastside Clinic'),
    ('S-107', 'DR-PATEL', '2026-03-05T10:30:00', 'Eastside Clinic'),
    ('S-108', 'DR-PATEL', '2026-03-10T15:00:00', 'Eastside Clinic'),
    ('S-109', 'DR-LEE', '2026-03-04T09:30:00', 'Main Clinic'),
    ('S-110', 'DR-LEE', '2026-03-11T10:00:00', 'Main Clinic'),
    ('S-111', 'DR-MORALES', '2026-03-05T16:00:00', 'Eastside Clinic'),
    ('S-112', 'DR-MORALES', '2026-03-12T09:00:00', 'Eastside Clinic');

INSERT INTO appointments (appointment_id, patient_id, slot_id, reason, status, cancel_reason) VALUES
    ('A-5001', 'P-1005', 'S-106', 'Blood pressure follow-up', 'booked', NULL);

INSERT INTO insurance (patient_id, payer, member_id, group_number) VALUES
    ('P-1001', 'Aetna', 'AET123456', 'G-100');
