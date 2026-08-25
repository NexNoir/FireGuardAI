import sqlite3
import config

conn = sqlite3.connect(config.DB_PATH)

print("\n=== DATABASE ===")
print(config.DB_PATH)

print("\n=== COLUMNS ===")

columns = conn.execute(
    "PRAGMA table_info(sensor_readings)"
).fetchall()

for column in columns:
    print(column)

print("\n=== ROW COUNT ===")

count = conn.execute(
    "SELECT COUNT(*) FROM sensor_readings"
).fetchone()[0]

print(count)

print("\n=== SAMPLE DATA ===")

rows = conn.execute(
    "SELECT * FROM sensor_readings LIMIT 5"
).fetchall()

for row in rows:
    print(row)

conn.close()