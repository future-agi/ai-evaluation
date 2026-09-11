CREATE DATABASE IF NOT EXISTS voice;
CREATE TABLE IF NOT EXISTS voice.calls (id UInt32, city String) ENGINE=MergeTree ORDER BY id;
INSERT INTO voice.calls VALUES (1, 'Bengaluru');
