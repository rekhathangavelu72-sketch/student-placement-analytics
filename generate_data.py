"""Create a synthetic placement.db (SQLite). Replace with real CSV loads for real data."""
import sqlite3
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 600

students = pd.DataFrame({
    "student_id": range(1, N + 1),
    "branch": rng.choice(["CSE", "IT", "ECE", "MECH", "CIVIL"], N, p=[.3, .2, .2, .15, .15]),
    "cgpa": rng.normal(7.3, 1.0, N).clip(5, 10).round(2),
    "backlogs": rng.poisson(0.5, N),
    "python": rng.integers(1, 11, N),
    "sql": rng.integers(1, 11, N),
    "communication": rng.integers(1, 11, N),
    "projects": rng.poisson(2, N),
    "certifications": rng.poisson(1.5, N),
})

# Internships: ~55% of students have at least one
rows = []
for sid in students.student_id:
    for _ in range(rng.choice([0, 1, 2], p=[.45, .38, .17])):
        rows.append((sid, rng.choice(["Startup", "MNC", "Research"], p=[.45, .4, .15]),
                     int(rng.choice([1, 2, 3, 6]))))
internships = pd.DataFrame(rows, columns=["student_id", "company_type", "duration_months"])
internships.insert(0, "internship_id", range(1, len(internships) + 1))

# Placement outcome driven by marks, skills, internships, backlogs (+ noise)
has_intern = students.student_id.isin(internships.student_id).astype(int)
skill_avg = students[["python", "sql", "communication"]].mean(axis=1)
logit = (-6.2 + 0.55 * students.cgpa + 1.0 * has_intern + 0.3 * skill_avg
         + 0.15 * students.projects - 0.8 * students.backlogs)
placed = rng.random(N) < 1 / (1 + np.exp(-logit))
package = np.where(placed, (3 + 0.8 * (students.cgpa - 6) + 0.2 * skill_avg
                            + rng.normal(0, 1.2, N)).clip(2.5, 25).round(1), np.nan)
placements = pd.DataFrame({"student_id": students.student_id,
                           "placed": placed.astype(int), "package_lpa": package})

with sqlite3.connect("placement.db") as con:
    students.to_sql("students", con, if_exists="replace", index=False)
    internships.to_sql("internships", con, if_exists="replace", index=False)
    placements.to_sql("placements", con, if_exists="replace", index=False)
    con.executescript("""
        DROP VIEW IF EXISTS student_profile;
        CREATE VIEW student_profile AS
        SELECT s.*, p.placed, p.package_lpa,
               COUNT(i.internship_id) AS internship_count
        FROM students s
        JOIN placements p ON p.student_id = s.student_id
        LEFT JOIN internships i ON i.student_id = s.student_id
        GROUP BY s.student_id;
    """)
print(f"placement.db created: {N} students, {len(internships)} internships")
