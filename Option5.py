# save as insider_study_app.py
import streamlit as st
import streamlit.components.v1 as components  # Required for scrolling JS
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import io
import os
import glob
from typing import Optional

# --- PDF GENERATION IMPORTS ---
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER

# -------------------- CONFIG / UTILITIES --------------------
APP_TITLE = "Insider Threat Study"
st.set_page_config(page_title=APP_TITLE, page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")

TORONTO_TZ = ZoneInfo("America/Toronto")

# DB filenames
DB_RESPONSES = "student_responses.db"
DB_FEEDBACK = "feedback.db"
DB_TRUST = "trust.db"
DB_QUESTIONNAIRE = "questionnaire.db"
GROUND_TRUTH_CSV = "ground_truth.csv"
DICTIONARY_SCENARIO_CSV = "dictionary_scenario.csv"

def now_toronto():
    return datetime.now(TORONTO_TZ)

def format_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def scroll_to_top():
    """Injects JS to scroll the main container to the top."""
    js = """
    <script>
        var body = window.parent.document.querySelector(".main");
        if (body) {
            body.scrollTop = 0;
        }
    </script>
    """
    components.html(js, height=0)

# -------------------- DB SCHEMAS --------------------
DB_FILES = {
    "feedback.db": '''
    CREATE TABLE IF NOT EXISTS questionnaire_phase (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        user_number INTEGER,
        phase INTEGER,
        q1 INTEGER, q2 INTEGER, q3 INTEGER, q4 INTEGER, q5 INTEGER,
        q6 INTEGER, q7 INTEGER, q8 INTEGER, q9 INTEGER, q10 TEXT,
        timestamp TEXT
    );
    ''',
    "questionnaire.db": '''
    CREATE TABLE IF NOT EXISTS analyst_info (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        user_number INTEGER,
        name TEXT, email TEXT, timestamp TEXT
    );
    ''',
    "trust.db": '''
    CREATE TABLE IF NOT EXISTS trust_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        user_number INTEGER,
        phase INTEGER,
        objective_trust REAL, subjective_trust REAL, combined_trust REAL,
        timestamp TEXT
    );
    ''',
    "student_responses.db": '''
    CREATE TABLE IF NOT EXISTS responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        user_number INTEGER,
        scenario INTEGER,
        answer TEXT, evidence TEXT, case_type TEXT,
        marks INTEGER, confidence_marks INTEGER,
        phase_time REAL, total_time REAL, timestamp TEXT
    );
    '''
}

# -------------------- SESSION STATE FLAGS --------------------
if "db_initialized" not in st.session_state:
    st.session_state["db_initialized"] = False
if "trigger_scroll" not in st.session_state:
    st.session_state["trigger_scroll"] = False

for db_name in DB_FILES.keys():
    flag = db_name.replace(".", "_") + "_submitted"
    if flag not in st.session_state:
        st.session_state[flag] = False

# Report bytes (only show download when present)
if "latest_report_pdf_bytes" not in st.session_state:
    st.session_state["latest_report_pdf_bytes"] = None
if "latest_report_filename" not in st.session_state:
    st.session_state["latest_report_filename"] = None

# -------------------- DB HELPER FUNCTIONS --------------------
def _init_single_db_file(db_path: str, schema_sql: str):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    tables = c.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    for (t,) in tables:
        if t == "sqlite_sequence":
            continue
        c.execute(f"DROP TABLE IF EXISTS {t};")
    c.executescript(schema_sql)
    conn.commit()
    conn.close()

def initialize_all_databases():
    for db_path, schema in DB_FILES.items():
        _init_single_db_file(db_path, schema)
    for db_name in DB_FILES.keys():
        flag = db_name.replace(".", "_") + "_submitted"
        st.session_state[flag] = False
    st.session_state["db_initialized"] = True
    st.success("Databases re-initialized.")

def create_tables_if_missing():
    for db_path, schema in DB_FILES.items():
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            conn.executescript(schema)
            conn.commit()
        finally:
            conn.close()

# -------------------- PERSISTENCE HELPERS --------------------
def save_response(student_id, user_number, scenario, answer, evidence, case_type,
                  objective_marks, confidence_marks, phase_time, total_time):
    toronto_time = now_toronto().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_RESPONSES)
    c = conn.cursor()
    c.execute(
        """INSERT INTO responses
           (student_id,user_number,scenario,answer,evidence,case_type,marks,confidence_marks,phase_time,total_time,timestamp)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (int(student_id), int(user_number), int(scenario), answer, evidence, case_type,
         int(objective_marks) if objective_marks is not None else None,
         int(confidence_marks) if confidence_marks is not None else None,
         float(phase_time), float(total_time), toronto_time)
    )
    conn.commit()
    conn.close()
    st.session_state["student_responses_db_submitted"] = True

def save_questionnaire_phase(student_id, user_number, phase, answers):
    toronto_time = now_toronto().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_FEEDBACK)
    c = conn.cursor()
    c.execute(
        """INSERT INTO questionnaire_phase
           (student_id,user_number,phase,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,timestamp)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (int(student_id), int(user_number), int(phase), *answers, toronto_time)
    )
    conn.commit()
    conn.close()
    st.session_state["feedback_db_submitted"] = True

def save_trust_score(student_id, user_number, phase, objective_trust, subjective_trust, combined_trust):
    toronto_time = now_toronto().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_TRUST)
    c = conn.cursor()
    c.execute(
        """INSERT INTO trust_scores
           (student_id,user_number,phase,objective_trust,subjective_trust,combined_trust,timestamp)
           VALUES (?,?,?,?,?,?,?)""",
        (int(student_id), int(user_number), int(phase),
         float(objective_trust) if objective_trust is not None else None,
         float(subjective_trust) if subjective_trust is not None else None,
         float(combined_trust) if combined_trust is not None else None,
         toronto_time)
    )
    conn.commit()
    conn.close()
    st.session_state["trust_db_submitted"] = True

# -------------------- DATA LOADING --------------------
@st.cache_data
def load_ground_truth(path: str = GROUND_TRUTH_CSV) -> pd.DataFrame:
    # (Kept for compatibility with your earlier code; not used in current UI.)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data
def load_dictionary_scenario(path: str = DICTIONARY_SCENARIO_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["user_number", "user_name", "scenario"])
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    if "user_number" in df.columns:
        df["user_number"] = pd.to_numeric(df["user_number"], errors="coerce").astype("Int64")
    if "scenario" in df.columns:
        df["scenario"] = pd.to_numeric(df["scenario"], errors="coerce").astype("Int64")
    return df

@st.cache_data
def load_raw_input_csv(path: str) -> pd.DataFrame:
    """
    Robust parsing for CERT r5.2-like rows that may contain commas in the tail.
    Expected logical structure:
        event_type,event_id,timestamp,user,pc,<details with commas>
    """
    rows = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 5)  # keep tail commas in details
            if len(parts) < 6:
                parts += [""] * (6 - len(parts))
            rows.append(parts)

    cols = ["event_type", "event_id", "timestamp", "user", "pc", "details"]
    return pd.DataFrame(rows, columns=cols)

# -------------------- SCENARIO / TRUST LOGIC --------------------
SCENARIO_OPTIONS = [1, 2, 3, 4]
SCENARIO_LABELS = {
    1: "1 - Data Exfiltration via USB",
    2: "2 - Emailing Sensitive Files",
    3: "3 - Uploading to Cloud Storage",
    4: "4 - Working Unusual Hours + Data Transfer",
    -1: "N/A",
    0: "0 - Normal User (Not Selectable)",
}

def scenario_label(x):
    return SCENARIO_LABELS.get(int(x), str(x))

def compute_trust_scores(student_id, user_number, phase, ground_truth_df=None, dict_df=None):
    subj = None
    conn = sqlite3.connect(DB_FEEDBACK)
    qdf = pd.read_sql_query(
        "SELECT q1,q2,q3,q4,q5,q6,q7,q8,q9 FROM questionnaire_phase WHERE student_id=? AND user_number=? AND phase=?",
        conn, params=(student_id, user_number, phase)
    )
    conn.close()
    if not qdf.empty:
        vals = qdf.mean(axis=0).values.astype(float)
        subj = round(float(vals.mean() * 20.0), 2)

    case_label = {1: "Without Explainability", 2: "With Explainability"}.get(phase)
    user_response = None
    if case_label:
        conn = sqlite3.connect(DB_RESPONSES)
        rdf = pd.read_sql_query(
            "SELECT scenario, answer FROM responses WHERE student_id=? AND user_number=? AND case_type=? ORDER BY timestamp DESC LIMIT 1",
            conn, params=(student_id, user_number, case_label)
        )
        conn.close()
        if not rdf.empty:
            user_response = rdf.iloc[0].to_dict()

    if dict_df is None:
        dict_df = load_dictionary_scenario()

    objective_score = None
    if user_response and dict_df is not None and not dict_df.empty:
        try:
            correct_row = dict_df[dict_df["user_number"] == int(user_number)]
            if not correct_row.empty:
                correct_scenario = int(correct_row.iloc[0]["scenario"])
                submitted_scenario = int(user_response.get("scenario", -1))
                scenario_marks = 50 if submitted_scenario == correct_scenario else 0
                correct_insider = (correct_scenario != 0)

                user_answer = str(user_response.get("answer", "")).strip().lower()
                user_insider = True if user_answer == "yes" else False if user_answer == "no" else None

                insider_marks = 0
                if user_insider is not None:
                    insider_marks = 50 if (user_insider == correct_insider) else 0

                objective_score = round(float(scenario_marks + insider_marks), 2)
        except Exception:
            objective_score = None

    combined = None
    if subj is not None and objective_score is not None:
        combined = round(float(subj + objective_score), 2)
    elif subj is None:
        combined = objective_score
    elif objective_score is None:
        combined = subj

    return {"subjective": subj, "objective": objective_score, "combined": combined}

# -------------------- REPORT GENERATION --------------------
def build_pdf_report_bytes(student_id: Optional[int], user_id: Optional[int]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "MainTitle",
        parent=styles["Heading1"],
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=18,
        textColor=colors.darkblue
    )

    normal = styles["Normal"]

    elements = []
    elements.append(Paragraph("Insider Threat Assessment Report", title_style))
    elements.append(Paragraph(f"Generated on: {now_toronto().strftime('%Y-%m-%d %H:%M')}", normal))
    elements.append(Spacer(1, 0.25 * inch))

    info_data = [
        ["Participant ID", str(student_id) if student_id is not None else "N/A"],
        ["Suspect User ID", str(user_id) if user_id is not None else "N/A"],
        ["Assessment Date", now_toronto().strftime("%Y-%m-%d")],
    ]
    info_table = Table(info_data, colWidths=[2 * inch, 3.5 * inch], hAlign="LEFT")
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.25 * inch))

    elements.append(Paragraph(
        "This report confirms successful report generation from the Insider Threat Study application. "
        "It can be extended to include comparative metrics (Design A vs Design B) from your databases if required.",
        normal
    ))

    doc.build(elements)
    return buffer.getvalue()

# ==================== PHASE/DESIGN UI ====================
def phase_ui_common(phase_num: int, dict_df: pd.DataFrame):
    design_label = "Design A" if phase_num == 1 else "Design B"
    st.header(design_label)

    # --- Input Section ---
    c1, c2, c3 = st.columns(3)
    with c1:
        default_student = st.session_state.get("student_id", 1)
        student_id = st.number_input(
            "Participant ID", min_value=1, max_value=9999, value=int(default_student), key=f"sid_p{phase_num}"
        )
        st.session_state["student_id"] = int(student_id)

    with c2:
        user_number = st.number_input(
            "Suspect User ID", min_value=0, max_value=9999, value=0, key=f"user_p{phase_num}"
        )
        st.session_state[f"phase{phase_num}_user"] = int(user_number)

    with c3:
        username_display = ""
        if dict_df is not None and not dict_df.empty:
            matched = dict_df[dict_df["user_number"] == int(user_number)]
            if not matched.empty:
                username_display = str(matched.iloc[0].get("user_name", ""))
        st.text_input("User Name", value=username_display, disabled=True, key=f"username_p{phase_num}")

    st.divider()

    # --- Timer ---
    if not st.session_state.get(f"phase{phase_num}_done", False):
        st.session_state[f"p{phase_num}_start"] = time.time()
        st.session_state["total_start"] = st.session_state.get("total_start", time.time())
    else:
        saved_phase_time = st.session_state.get(f"phase{phase_num}_time", None)
        if saved_phase_time is not None:
            st.success(f"{design_label} completed. Recorded time: {format_time(saved_phase_time)}")

    # --- Evidence ---
    st.subheader("Analysis Data & Evidence")

    # Anomaly loss only (ROC removed)
    st.markdown("**1. Anomaly Loss**")
    loss_path = f"./user_{int(user_number):03d}/user_{int(user_number):03d}_anomaly_scores.png"
    if os.path.exists(loss_path):
        st.image(loss_path, use_container_width=True)
    else:
        st.warning("Image not found.")

    st.markdown("---")
    st.markdown("**2. CDF Curve**")
    scores_path = f"./user_{int(user_number):03d}/user_{int(user_number):03d}_cdf_curve.png"
    if os.path.exists(scores_path):
        st.image(scores_path, use_container_width=True)
    else:
        st.info("Image not available.")

    # --- Input & Output Data ---
    st.markdown("---")
    st.markdown("**3. Model Input and Output Data**")

    with st.expander("View Input Data (Click to Expand)"):
        if not username_display:
            st.info("No username found for the selected Suspect User ID.")
        else:
            folders = ["r5.2-1", "r5.2-2", "r5.2-3", "r5.2-4"]
            matches = []
            for folder in folders:
                if os.path.isdir(folder):
                    pattern = os.path.join(folder, f"*{username_display}*.csv")
                    matches.extend(glob.glob(pattern))

            if not matches:
                st.warning(f"No CSV files found for user '{username_display}' in r5.2-1..4.")
            else:
                if len(matches) == 1:
                    chosen_path = matches[0]
                    st.write(f"Loaded file: `{os.path.basename(chosen_path)}`")
                else:
                    chosen_path = st.selectbox(
                        "Multiple CSV files found. Select one:",
                        options=matches,
                        format_func=lambda p: os.path.join(os.path.basename(os.path.dirname(p)), os.path.basename(p)),
                        key=f"raw_csv_select_p{phase_num}"
                    )

                try:
                    raw_df = load_raw_input_csv(chosen_path)
                    st.dataframe(raw_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to read CSV file: {e}")

    with st.expander("View Output Data (Click to Expand)"):
        log_file_path = "Output2000"
        try:
            with open(log_file_path, "r") as f:
                log_content = f.read()

            testing_index = log_content.find("Testing...")
            if testing_index != -1:
                log_content = log_content[testing_index:]

            import re
            pattern_re = re.compile(
                r"\[(\d+)\] DAY (\d{4}-\d{2}-\d{2}) : ([\d.]+) => \(ans,pred\) (True|False):(True|False)\]"
            )

            entries = []
            for line in log_content.splitlines():
                match = pattern_re.match(line)
                if match:
                    entries.append({
                        "User": int(match.group(1)),
                        "Date": match.group(2),
                        "Loss": float(match.group(3)),
                        "Ans": match.group(4) == "True",
                        "Pred": match.group(5) == "True",
                    })

            if not entries:
                st.info("No logs found.")
            else:
                df_log = pd.DataFrame(entries)
                df_user = df_log[df_log["User"] == int(user_number)].reset_index(drop=True)

                if df_user.empty:
                    st.info("No output data for this user.")
                else:
                    df_view = df_user.drop(columns=["Ans"])
                    df_view = df_view.rename(columns={"Pred": "Behavioral Deviation"})
                    st.dataframe(df_view.style.format({"Loss": "{:.6f}"}), use_container_width=True)

        except Exception:
            st.info("Log file unreadable.")

    # SHAP only in Design B
    if phase_num == 2:
        st.markdown("---")
        st.subheader("Explainability (SHAP)")
        col_bee, col_water = st.columns(2)
        with col_bee:
            st.markdown("**SHAP Beeswarm**")
            path = f"./user_{int(user_number):03d}/user_{int(user_number):03d}_shap_beeswarm_test.png"
            if os.path.exists(path):
                st.image(path, use_container_width=True)
        with col_water:
            st.markdown("**SHAP Waterfall**")
            pattern = f"./user_{int(user_number):03d}/user_{int(user_number):03d}_shap_waterfall_*.png"
            files = sorted(glob.glob(pattern))
            if files:
                for fpath in files:
                    st.image(fpath, use_container_width=True)

    st.divider()

    # --- Decision ---
    st.subheader("Assessment")
    col_dec1, col_dec2 = st.columns(2)

    conf_labels = {
        1: "No confidence",
        2: "Little confidence",
        3: "Mid-level confidence",
        4: "Higher confidence",
        5: "Highest confidence",
    }

    with col_dec1:
        decision = st.radio(
            "Is this a Potential Insider Threat?",
            ["No", "Yes"],
            index=0,
            key=f"p{phase_num}_decision",
            horizontal=True,
        )

        # Scenario dropdown is visible ONLY when decision is Yes
        if decision == "Yes":
            scenario = st.selectbox(
                "Select Scenario Type",
                SCENARIO_OPTIONS,
                format_func=scenario_label,
                index=0,
                key=f"p{phase_num}_scenario_yes",
            )
        else:
            scenario = -1

    with col_dec2:
        decision_confidence = st.select_slider(
            "Decision Confidence",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: f"{x} - {conf_labels[x]}",
            key=f"p{phase_num}_decision_conf",
        )
        scenario_confidence = st.select_slider(
            "Scenario Confidence",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: f"{x} - {conf_labels[x]}",
            key=f"p{phase_num}_scenario_conf",
        )

    submit_btn = st.button(f"Submit {design_label}", key=f"submit_p{phase_num}", type="primary")

    if submit_btn and not st.session_state.get(f"phase{phase_num}_done", False):
        phase_time = time.time() - st.session_state.get(f"p{phase_num}_start", time.time())
        total_time = time.time() - st.session_state.get("total_start", time.time())
        st.session_state[f"phase{phase_num}_time"] = phase_time
        st.session_state[f"phase{phase_num}_total_time"] = total_time

        # Objective marks
        obj_marks = None
        try:
            correct_row = dict_df[dict_df["user_number"] == int(user_number)]
            if not correct_row.empty:
                correct_scenario = int(correct_row.iloc[0]["scenario"])
                scenario_marks = 50 if int(scenario) == correct_scenario else 0
                correct_insider = (correct_scenario != 0)
                user_insider = True if decision.lower() == "yes" else False if decision.lower() == "no" else None
                insider_marks = 50 if (user_insider is not None and user_insider == correct_insider) else 0
                obj_marks = scenario_marks + insider_marks
        except Exception:
            obj_marks = None

        # Confidence marks (kept consistent with your earlier design)
        confidence_marks = ((decision_confidence - 1) + (scenario_confidence - 1)) / 2.0 * 25.0
        confidence_text = f"Decision confidence: {decision_confidence}/5; Scenario confidence: {scenario_confidence}/5."

        case_type = "Without Explainability" if phase_num == 1 else "With Explainability"

        try:
            save_response(
                student_id=int(st.session_state["student_id"]),
                user_number=int(user_number),
                scenario=int(scenario),
                answer=decision,
                evidence=confidence_text,
                case_type=case_type,
                objective_marks=obj_marks,
                confidence_marks=confidence_marks,
                phase_time=phase_time,
                total_time=total_time,
            )
            st.session_state[f"phase{phase_num}_done"] = True
            st.success(f"{design_label} Submitted!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    # --- Questionnaire ---
    if st.session_state.get(f"phase{phase_num}_done", False) and not st.session_state.get(f"questionnaire_p{phase_num}_done", False):
        st.markdown("---")
        st.subheader(f"Questionnaire ({design_label})")
        st.info("Rate the following (1=Strongly Disagree, 5=Strongly Agree)")

        def radio_question(q, key):
            return st.radio(q, [1, 2, 3, 4, 5], index=2, key=key, horizontal=True)

        c_q1, c_q2 = st.columns(2)
        with c_q1:
            q1 = radio_question("1. The system was easy to use.", f"q1_p{phase_num}")
            q2 = radio_question("2. I trusted the AI's prediction.", f"q2_p{phase_num}")
            q3 = radio_question("3. The explanations helped me make a better decision.", f"q3_p{phase_num}")
            q4 = radio_question("4. I would use this system in real life.", f"q4_p{phase_num}")
            q5 = radio_question("5. The visualizations were clear.", f"q5_p{phase_num}")
        with c_q2:
            q6 = radio_question("6. I felt confident in my final answer.", f"q6_p{phase_num}")
            q7 = radio_question("7. The timer helped me focus.", f"q7_p{phase_num}")
            q8 = radio_question("8. I understood the loss values.", f"q8_p{phase_num}")
            q9 = radio_question("9. The CDF curve was useful.", f"q9_p{phase_num}")

        q10 = st.text_area("10. Comments?", height=80, key=f"q10_p{phase_num}")

        if st.button("Submit Questionnaire", key=f"submit_q_p{phase_num}"):
            answers = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]
            save_questionnaire_phase(int(st.session_state["student_id"]), int(user_number), phase_num, answers)

            trust = compute_trust_scores(int(st.session_state["student_id"]), int(user_number), phase_num, None, dict_df)
            try:
                save_trust_score(int(st.session_state["student_id"]), int(user_number), phase_num,
                                 trust["objective"], trust["subjective"], trust["combined"])
                st.session_state[f"questionnaire_p{phase_num}_done"] = True
                st.session_state["trigger_scroll"] = True
                st.success("Questionnaire Submitted!")
                st.rerun()
            except Exception as e:
                st.warning(f"Error: {e}")

# ==================== MAIN ====================
def main():
    if st.session_state.get("trigger_scroll", False):
        scroll_to_top()
        st.session_state["trigger_scroll"] = False

    st.title(APP_TITLE)
    page_choice = st.radio("Navigation:", ["Instructions", "Design A", "Design B"], index=0, horizontal=True)

    # Always ensure tables exist
    create_tables_if_missing()
    dict_df = load_dictionary_scenario()

    with st.sidebar:
        st.header("Admin")
        if st.button("Initialize DBs"):
            initialize_all_databases()

        with st.expander("📥 Download DBs"):
            for db in DB_FILES.keys():
                if os.path.exists(db):
                    with open(db, "rb") as f:
                        st.download_button(db, f, file_name=db)

        st.header("Reports")
        generate_report = st.button("Create Professional PDF Report")

        # Show download button ONLY after report is created
        if st.session_state.get("latest_report_pdf_bytes") is not None:
            st.download_button(
                "Download Report (PDF)",
                data=st.session_state["latest_report_pdf_bytes"],
                file_name=st.session_state.get("latest_report_filename", "Report.pdf"),
                mime="application/pdf",
            )

    # Create report on click (and only then enable download)
    if generate_report:
        sid = st.session_state.get("student_id", None)
        u1 = st.session_state.get("phase1_user", None)
        u2 = st.session_state.get("phase2_user", None)
        display_user = u1 if u1 is not None else u2

        try:
            pdf_bytes = build_pdf_report_bytes(sid, display_user)
            st.session_state["latest_report_pdf_bytes"] = pdf_bytes
            sid_str = str(sid) if sid is not None else "NA"
            st.session_state["latest_report_filename"] = f"Report_{sid_str}.pdf"
            st.sidebar.success("Report created. Download button is now available.")
            st.rerun()
        except Exception as e:
            st.session_state["latest_report_pdf_bytes"] = None
            st.session_state["latest_report_filename"] = None
            st.sidebar.error(f"Report generation failed: {e}")

    if page_choice == "Instructions":
        st.markdown(
            "## Instructions\n"
            "1. Enter Participant ID & User ID.\n"
            "2. For each design (A and B), review the evidence (loss, CDF, input and output data).\n"
            "3. Make your decision, select scenario (only if Yes) and confidence.\n"
            "4. Submit and then complete the questionnaire."
        )
    elif page_choice == "Design A":
        phase_ui_common(1, dict_df)
    elif page_choice == "Design B":
        phase_ui_common(2, dict_df)

if __name__ == "__main__":
    main()
