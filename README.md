# Student Placement Analytics System

Analyses marks, skills, internships and placement outcomes to find what is associated with getting placed.

**Stack:** Python · Pandas · SQL (SQLite) · Streamlit · scikit-learn

## Run
```bash
pip install -r requirements.txt
python generate_data.py     # builds placement.db (synthetic data)
streamlit run app.py
```

## Structure
- `generate_data.py` – creates `students`, `internships`, `placements` tables and a `student_profile` view.
- `app.py` – dashboard: filters, KPIs, academics / skills / internships tabs, and a logistic-regression driver chart.

## Using real data
Load your CSVs into the same three tables (see column names in `generate_data.py`); the app needs no changes.

## Ideas to extend
Add a "predict my chances" form, cross-validated model metrics, salary-band analysis, CSV export.
