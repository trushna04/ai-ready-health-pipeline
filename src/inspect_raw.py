"""
inspect_raw.py  (Phase 0)

Profiles the raw CSVs with plain SQL in DuckDB, to PROVE the mess is real
before we try to clean it. It reads ONLY data/raw/, never the answer key.

Run:
  python src/inspect_raw.py
"""

import duckdb

con = duckdb.connect()  # in-memory database; nothing is saved to disk

# all_varchar=true: load every column as TEXT.
# If we let DuckDB guess types, it would quietly turn "N/A" into NULL,
# "00412345" into the number 412345, and choke on mixed date formats.
# Raw data should be loaded exactly as it arrived; parsing is Phase 1's job.
con.sql("CREATE TABLE patients   AS SELECT * FROM read_csv('data/raw/patients.csv',   all_varchar=true)")
con.sql("CREATE TABLE encounters AS SELECT * FROM read_csv('data/raw/encounters.csv', all_varchar=true)")


def show(title, query):
    print(f"\n=== {title} ===")
    con.sql(query).show(max_rows=12)


show("1. Row counts by source system", """
    SELECT p.source_system, p.patient_rows, e.encounter_rows
    FROM (SELECT source_system, COUNT(*) AS patient_rows FROM patients GROUP BY 1) p
    JOIN (SELECT source_system, COUNT(*) AS encounter_rows FROM encounters GROUP BY 1) e
      USING (source_system)
    ORDER BY 1
""")

show("2. Exact duplicate rows (same row loaded twice)", """
    SELECT 'patients' AS tbl, COUNT(*) - (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM patients)) AS extra_copies FROM patients
    UNION ALL
    SELECT 'encounters', COUNT(*) - (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM encounters)) FROM encounters
""")

# Shape trick: replace every digit with 9 and every letter with A.
# '3/14/85' -> '9/99/99', 'March 14, 1985' -> 'AAAAA 99, 9999'.
# Each distinct shape is a different date format we must handle.
show("3. DOB formats (digits->9, letters->A)", """
    SELECT regexp_replace(regexp_replace(dob, '[0-9]', '9', 'g'), '[A-Za-z]', 'A', 'g') AS dob_shape,
           COUNT(*) AS n
    FROM patients GROUP BY dob_shape ORDER BY n DESC
""")

# You'll see NULL twice: one is the literal TEXT "NULL" someone typed,
# the other is a real NULL (DuckDB turns an empty field "" into NULL even as text).
show("4. All the ways 'missing' is spelled (sex column)", """
    SELECT sex, COUNT(*) AS n FROM patients GROUP BY sex ORDER BY n DESC
""")

# Same last name + first initial in more than one record = candidate duplicate.
# Crude on purpose: it shows the size of the matching problem, not the answer.
show("5. Possible same person, multiple records (crude: last name + first initial)", """
    SELECT upper(last_name) AS last_name, upper(left(first_name, 1)) AS first_initial,
           COUNT(DISTINCT source_record_id) AS records,
           string_agg(DISTINCT first_name || ' @' || source_system, ', ') AS seen_as
    FROM patients
    GROUP BY 1, 2 HAVING COUNT(DISTINCT source_record_id) > 1
    ORDER BY records DESC
""")

# Orphans: encounters whose MRN is not in the patients table.
# Some are truly lost; some are clinic_b MRNs that lost their leading zeros.
show("6. Orphan encounters (MRN not found in patients)", """
    SELECT e.source_system, COUNT(*) AS orphan_rows,
           COUNT(*) FILTER (WHERE lpad(e.patient_mrn, 8, '0') IN (SELECT mrn FROM patients)) AS fixable_by_zero_padding
    FROM encounters e
    WHERE e.patient_mrn NOT IN (SELECT mrn FROM patients)
    GROUP BY 1 ORDER BY 1
""")

# try_cast returns NULL instead of crashing when the text isn't a number.
show("7. Impossible blood pressure values", """
    SELECT systolic_bp, COUNT(*) AS n FROM encounters
    WHERE try_cast(systolic_bp AS INTEGER) NOT BETWEEN 50 AND 300
    GROUP BY 1 ORDER BY n DESC
""")

show("8. PHI leaked into free-text notes (phone, MRN, or names)", """
    SELECT encounter_id, diagnosis_note FROM encounters
    WHERE regexp_matches(diagnosis_note, '\\d{3}-555-01\\d{2}|MRN|spouse')
    LIMIT 5
""")
