import sqlite3
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Placement Analytics", layout="wide")
DB = "placement.db"


@st.cache_data
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    with sqlite3.connect(DB) as con:
        return pd.read_sql_query(sql, con, params=params)


# ---- Sidebar filters (applied in SQL via parameters) ----
branches = query("SELECT DISTINCT branch FROM students ORDER BY 1").branch.tolist()
sel = st.sidebar.multiselect("Branch", branches, default=branches)
cg = st.sidebar.slider("CGPA range", 5.0, 10.0, (5.0, 10.0), 0.1)
if not sel:
    st.warning("Select at least one branch.")
    st.stop()

marks = ",".join("?" * len(sel))
WHERE = f"WHERE branch IN ({marks}) AND cgpa BETWEEN ? AND ?"
P = (*sel, *cg)
df = query(f"SELECT * FROM student_profile {WHERE}", P)

st.title("🎓 Student Placement Analytics")
if df.empty:
    st.info("No students match these filters.")
    st.stop()

# ---- KPIs ----
placed = df[df.placed == 1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Students", len(df))
c2.metric("Placement rate", f"{df.placed.mean():.1%}")
c3.metric("Avg package (LPA)", f"{placed.package_lpa.mean():.1f}" if len(placed) else "–")
c4.metric("With internship", f"{(df.internship_count > 0).mean():.1%}")

tab1, tab2, tab3, tab4 = st.tabs(["Academics", "Skills", "Internships", "Key drivers"])

with tab1:
    a, b = st.columns(2)
    by_branch = query(f"""SELECT branch, ROUND(AVG(placed)*100,1) AS placement_pct
                          FROM student_profile {WHERE} GROUP BY branch""", P)
    a.subheader("Placement % by branch")
    a.bar_chart(by_branch, x="branch", y="placement_pct")
    bands = pd.cut(df.cgpa, [0, 6, 7, 8, 9, 10], labels=["<6", "6-7", "7-8", "8-9", "9-10"])
    b.subheader("Placement % by CGPA band")
    b.bar_chart((df.groupby(bands, observed=True).placed.mean() * 100).round(1))
    st.subheader("Placement % by number of backlogs")
    st.bar_chart((df.groupby(df.backlogs.clip(upper=3)).placed.mean() * 100).round(1))

with tab2:
    skill = st.selectbox("Skill", ["python", "sql", "communication"])
    lvl = pd.cut(df[skill], [0, 3, 6, 10], labels=["Low (1-3)", "Medium (4-6)", "High (7-10)"])
    st.subheader(f"Placement % by {skill} level")
    st.bar_chart((df.groupby(lvl, observed=True).placed.mean() * 100).round(1))
    st.subheader("Projects & certifications vs placement")
    st.dataframe(df.groupby("projects").agg(students=("placed", "size"),
                                            placement_rate=("placed", "mean")).round(2))

with tab3:
    st.subheader("Internship count vs outcome (SQL)")
    st.dataframe(query(f"""
        SELECT internship_count, COUNT(*) AS students,
               ROUND(AVG(placed)*100,1) AS placement_pct,
               ROUND(AVG(package_lpa),2) AS avg_package_lpa
        FROM student_profile {WHERE}
        GROUP BY internship_count ORDER BY internship_count""", P), hide_index=True)
    st.subheader("Placement % by internship company type (SQL)")
    st.dataframe(query(f"""
        SELECT i.company_type, COUNT(DISTINCT sp.student_id) AS students,
               ROUND(AVG(sp.placed)*100,1) AS placement_pct
        FROM internships i JOIN student_profile sp USING (student_id)
        {WHERE.replace('branch', 'sp.branch').replace('cgpa', 'sp.cgpa')}
        GROUP BY i.company_type""", P), hide_index=True)

with tab4:
    st.subheader("What predicts placement? (standardised logistic regression)")
    feats = ["cgpa", "backlogs", "python", "sql", "communication",
             "projects", "certifications", "internship_count"]
    if df.placed.nunique() < 2:
        st.info("Need both placed and unplaced students to fit a model.")
    else:
        X = StandardScaler().fit_transform(df[feats])
        model = LogisticRegression(max_iter=1000).fit(X, df.placed)
        coef = pd.Series(model.coef_[0], index=feats).sort_values()
        st.bar_chart(coef)
        st.caption("Positive = raises placement odds, negative = lowers them (per 1 std. dev.). "
                   "These are associations in this data, not proof of cause.")
