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
import seaborn as sns

# --- PDF GENERATION IMPORTS ---
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# -------------------- CONFIG / UTILITIES --------------------
APP_TITLE = "Insider Threat Study"

# Set page config
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

# --- SCROLL TO TOP HELPER ---
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
if 'db_initialized' not in st.session_state:
    st.session_state['db_initialized'] = False
if 'trigger_scroll' not in st.session_state:
    st.session_state['trigger_scroll'] = False

for db_name in DB_FILES.keys():
    flag = db_name.replace('.', '_') + "_submitted"
    if flag not in st.session_state:
        st.session_state[flag] = False

# -------------------- DB HELPER FUNCTIONS --------------------
def _init_single_db_file(db_path, schema_sql):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    tables = c.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    for (t,) in tables:
        if t == "sqlite_sequence": continue
        c.execute(f"DROP TABLE IF EXISTS {t};")
    c.executescript(schema_sql)
    conn.commit()
    conn.close()

def initialize_all_databases():
    for db_path, schema in DB_FILES.items():
        _init_single_db_file(db_path, schema)
    for db_name in DB_FILES.keys():
        flag = db_name.replace('.', '_') + "_submitted"
        st.session_state[flag] = False
    st.session_state['db_initialized'] = True
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
def save_response(student_id, user_number, scenario, answer, evidence, case_type, objective_marks, confidence_marks, phase_time, total_time):
    toronto_time = now_toronto().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_RESPONSES)
    c = conn.cursor()
    c.execute(
        """INSERT INTO responses (student_id,user_number,scenario,answer,evidence,case_type,marks,confidence_marks,phase_time,total_time,timestamp)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (int(student_id), int(user_number), int(scenario), answer, evidence, case_type,
         int(objective_marks) if objective_marks is not None else None,
         int(confidence_marks) if confidence_marks is not None else None,
         float(phase_time), float(total_time), toronto_time)
    )
    conn.commit()
    conn.close()
    st.session_state['student_responses_db_submitted'] = True

def save_questionnaire_phase(student_id, user_number, phase, answers):
    toronto_time = now_toronto().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_FEEDBACK)
    c = conn.cursor()
    c.execute(
        """INSERT INTO questionnaire_phase (student_id,user_number,phase,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,timestamp)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (int(student_id), int(user_number), int(phase), *answers, toronto_time)
    )
    conn.commit()
    conn.close()
    st.session_state['feedback_db_submitted'] = True

def save_trust_score(student_id, user_number, phase, objective_trust, subjective_trust, combined_trust):
    toronto_time = now_toronto().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_TRUST)
    c = conn.cursor()
    c.execute(
        """INSERT INTO trust_scores (student_id,user_number,phase,objective_trust,subjective_trust,combined_trust,timestamp)
           VALUES (?,?,?,?,?,?,?)""",
        (int(student_id), int(user_number), int(phase),
         float(objective_trust) if objective_trust is not None else None,
         float(subjective_trust) if subjective_trust is not None else None,
         float(combined_trust) if combined_trust is not None else None,
         toronto_time)
    )
    conn.commit()
    conn.close()
    st.session_state['trust_db_submitted'] = True

# -------------------- DATA LOADING --------------------
@st.cache_data
def load_ground_truth(path=GROUND_TRUTH_CSV):
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data
def load_dictionary_scenario(path=DICTIONARY_SCENARIO_CSV):
    if not os.path.exists(path):
        return pd.DataFrame(columns=['user_number','user_name','scenario'])
    df = pd.read_csv(path, sep=None, engine='python')
    df.columns = [c.strip() for c in df.columns]
    if 'user_number' in df.columns:
        df['user_number'] = pd.to_numeric(df['user_number'], errors='coerce').astype('Int64')
    if 'scenario' in df.columns:
        df['scenario'] = pd.to_numeric(df['scenario'], errors='coerce').astype('Int64')
    return df

# -------------------- SCENARIO / TRUST LOGIC --------------------
SCENARIO_OPTIONS = [-1, 0, 1, 2, 3, 4, 5]
SCENARIO_LABELS = {
    -1: "I Don't Know",
     0: "0 - Normal User (No Threat)",
     1: "1 - Data Exfiltration via USB",
     2: "2 - Emailing Sensitive Files",
     3: "3 - Uploading to Cloud Storage",
     4: "4 - Working Unusual Hours + Data Transfer",
     5: "5 - I Don't Know"
}
def scenario_label(x):
    return SCENARIO_LABELS.get(int(x), str(x))

def _extract_confusion_for_user(ground_truth_df, user_number):
    if ground_truth_df is None or ground_truth_df.empty: return None
    possible_user_cols = ['user', 'user_number', 'user_id', 'userid', 'user number', 'id']
    user_col = next((c for c in ground_truth_df.columns if c.strip().lower() in possible_user_cols), ground_truth_df.columns[0])
    row = ground_truth_df[ground_truth_df[user_col] == user_number]
    if row.empty: return None
    row = row.iloc[0]
    mapping = {}
    for c in ground_truth_df.columns:
        cl = c.strip().lower()
        if cl == 'tp' or 'true positive' in cl: mapping['TP'] = c
        if cl == 'tn' or 'true negative' in cl: mapping['TN'] = c
        if cl == 'fp' or 'false positive' in cl: mapping['FP'] = c
        if cl == 'fn' or 'false negative' in cl: mapping['FN'] = c
    # fallback
    for key in ['TP','TN','FP','FN']:
        if key not in mapping:
            for c in ground_truth_df.columns:
                if c.strip().lower() == key.lower():
                    mapping[key] = c; break
    tp = int(row[mapping['TP']]) if 'TP' in mapping and pd.notna(row[mapping['TP']]) else 0
    tn = int(row[mapping['TN']]) if 'TN' in mapping and pd.notna(row[mapping['TN']]) else 0
    fp = int(row[mapping['FP']]) if 'FP' in mapping and pd.notna(row[mapping['FP']]) else 0
    fn = int(row[mapping['FN']]) if 'FN' in mapping and pd.notna(row[mapping['FN']]) else 0
    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}

def compute_trust_scores(student_id, user_number, phase, ground_truth_df=None, dict_df=None):
    subj = None
    obj = None
    # subjective
    conn = sqlite3.connect(DB_FEEDBACK)
    qdf = pd.read_sql_query(
        "SELECT q1,q2,q3,q4,q5,q6,q7,q8,q9 FROM questionnaire_phase WHERE student_id=? AND user_number=? AND phase=?",
        conn, params=(student_id, user_number, phase)
    )
    conn.close()
    if not qdf.empty:
        vals = qdf.mean(axis=0).values.astype(float)
        subj = round(float(vals.mean() * 20.0),2)

    # objective
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

    if dict_df is None: dict_df = load_dictionary_scenario()
    objective_score = None
    if user_response and dict_df is not None and not dict_df.empty:
        try:
            correct_row = dict_df[dict_df['user_number'] == int(user_number)]
            if not correct_row.empty:
                correct_scenario = int(correct_row.iloc[0]['scenario'])
                submitted_scenario = int(user_response.get('scenario', -1))
                scenario_marks = 50 if submitted_scenario == correct_scenario else 0
                correct_insider = (correct_scenario != 0)
                user_answer = str(user_response.get('answer', '')).strip().lower()
                user_insider = True if user_answer == 'yes' else False if user_answer == 'no' else None
                insider_marks = 0
                if user_insider is not None:
                    insider_marks = 50 if (user_insider == correct_insider) else 0
                objective_score = round(float(scenario_marks + insider_marks),2)
        except Exception: objective_score = None

    combined = None
    if subj is not None and objective_score is not None:
        combined = round(float(subj + objective_score),2)
    elif subj is None: combined = objective_score
    elif objective_score is None: combined = subj

    return {"subjective": subj, "objective": objective_score, "combined": combined}

# ==================== PHASE UI ====================
def phase_ui_common(phase_num, ground_truth_df, dict_df):
    st.header(f"Phase {phase_num} — {'Without Explainability' if phase_num==1 else 'With Explainability'}")
    
    # --- Input Section ---
    c1, c2, c3 = st.columns(3)
    with c1:
        default_student = st.session_state.get('student_id', 1)
        student_id = st.number_input("Participant ID", min_value=1, max_value=9999, value=int(default_student), key=f"sid_p{phase_num}")
        st.session_state['student_id'] = student_id
    with c2:
        user_number = st.number_input("Suspect User ID", min_value=0, max_value=9999, value=0, key=f"user_p{phase_num}")
        st.session_state[f'phase{phase_num}_user'] = user_number
    with c3:
        username_display = ""
        if dict_df is not None and not dict_df.empty:
            matched = dict_df[dict_df['user_number'] == int(user_number)]
            if not matched.empty: username_display = str(matched.iloc[0].get('user_name',''))
        st.text_input("User Name", value=username_display, disabled=True, key=f"username_p{phase_num}")

    st.divider()

    # --- Timer ---
    if not st.session_state.get(f'phase{phase_num}_done', False):
        st.session_state[f'p{phase_num}_start'] = time.time()
        st.session_state['total_start'] = st.session_state.get('total_start', time.time())
    else:
        saved_phase_time = st.session_state.get(f'phase{phase_num}_time', None)
        if saved_phase_time is not None:
            st.success(f"Phase completed. Recorded Phase time: {format_time(saved_phase_time)}")

    # --- Evidence (Grid) ---
    st.subheader("Analysis Data & Evidence")
    col_loss, col_roc = st.columns(2)
    with col_loss:
        st.markdown("**1. Anomaly Loss**")
        loss_path = f"./user_{user_number:03d}/user_{user_number:03d}_anomaly_scores.png"
        if os.path.exists(loss_path): st.image(loss_path, use_container_width=True)
        else: st.warning("Image not found.")
    with col_roc:
        st.markdown("**2. ROC Curve**")
        roc_path = f"./user_{user_number:03d}/user_{user_number:03d}_roc_curve.png"
        if os.path.exists(roc_path): st.image(roc_path, use_container_width=True)
        else: st.warning("Image not found.")
            
    st.markdown("---")
    col_conf, col_cdf = st.columns(2)
    with col_conf:
        st.markdown("**3. Confusion Matrix**")
        if not ground_truth_df.empty:
            conf_temp = _extract_confusion_for_user(ground_truth_df, user_number)
            if conf_temp:
                fig, ax = plt.subplots(figsize=(5,4))
                cm = [[conf_temp['TP'], conf_temp['FN']], [conf_temp['FP'], conf_temp['TN']]]
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred Yes","Pred No"], yticklabels=["Actual Yes","Actual No"], ax=ax)
                st.pyplot(fig)
            else: st.info("Data unavailable.")
        else: st.info("Ground truth missing.")
    with col_cdf:
        st.markdown("**4. CDF Curve**")
        scores_path = f"./user_{user_number:03d}/user_{user_number:03d}_cdf_curve.png"
        if os.path.exists(scores_path): st.image(scores_path, use_container_width=True)
        else: st.info("Image not available.")

    # Phase 2 SHAP
    if phase_num == 2:
        st.markdown("---")
        st.subheader("Explainability (SHAP)")
        col_bee, col_water = st.columns(2)
        with col_bee:
            st.markdown("**SHAP Beeswarm**")
            path = f"./user_{user_number:03d}/user_{user_number:03d}_shap_beeswarm_test.png"
            if os.path.exists(path): st.image(path, use_container_width=True)
        with col_water:
            st.markdown("**SHAP Waterfall**")
            pattern = f"./user_{user_number:03d}/user_{user_number:03d}_shap_waterfall_*.png"
            files = sorted(glob.glob(pattern))
            if files:
                for f in files: st.image(f, use_container_width=True)

    # Raw Logs
    with st.expander("View Raw Log Data (Click to Expand)"):
        #log_file_path = 'Output2000.log'
        log_file_path = 'Output2000.txt'
        try:
            with open(log_file_path, 'r') as f:
                log_content = f.read()
            testing_index = log_content.find("Testing...")
            if testing_index != -1:
                log_content = log_content[testing_index:]
                import re
                pattern_re = re.compile(r'\[(\d+)\] DAY (\d{4}-\d{2}-\d{2}) : ([\d.]+) => \(ans,pred\) (True|False):(True|False)\]')
                entries = []
                for line in log_content.splitlines():
                    match = pattern_re.match(line)
                    if match:
                        entries.append({'User': int(match.group(1)), 'Date': match.group(2), 'Loss': float(match.group(3)),
                                        'Ans': match.group(4)=='True', 'Pred': match.group(5)=='True'})
                if entries:
                    df_log = pd.DataFrame(entries)
                    st.dataframe(df_log[df_log['User'] == user_number].reset_index(drop=True), use_container_width=True)
                else: st.info("No logs found.")
        except: st.info("Log file unreadable.")

    st.divider()

    # --- Decision ---
    st.subheader("Assessment")
    col_dec1, col_dec2 = st.columns(2)
    with col_dec1:
        decision = st.radio("Is this an Insider Threat?", ["No","Yes"], index=0, key=f"p{phase_num}_decision", horizontal=True)
        if decision == "No":
            scenario = -1
            st.selectbox("Scenario Type", [-1], format_func=scenario_label, index=0, disabled=True, key=f"p{phase_num}_scenario_no")
        else:
            scenario = st.selectbox("Select Scenario Type", SCENARIO_OPTIONS[1:5]+[-1], format_func=scenario_label, index=0, key=f"p{phase_num}_scenario_yes")

    with col_dec2:
        confidence_score = st.select_slider(
            "Confidence Level",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: f"{x} - {'Not Confident' if x==1 else 'Very Confident' if x==5 else 'Somewhat Confident' if x==3 else ''}",
            key=f"p{phase_num}_confidence"
        )

    submit_btn = st.button(f"Submit Phase {phase_num}", key=f"submit_p{phase_num}", type="primary")

    if submit_btn and not st.session_state.get(f'phase{phase_num}_done', False):
        phase_time = time.time() - st.session_state.get(f'p{phase_num}_start', time.time())
        total_time = time.time() - st.session_state.get('total_start', time.time())
        st.session_state[f'phase{phase_num}_time'] = phase_time
        st.session_state[f'phase{phase_num}_total_time'] = total_time

        # Marks Calculation
        obj_marks = None
        try:
            correct_row = dict_df[dict_df['user_number'] == int(user_number)]
            if not correct_row.empty:
                correct_scenario = int(correct_row.iloc[0]['scenario'])
                scenario_marks = 50 if int(scenario) == correct_scenario else 0
                correct_insider = (correct_scenario != 0)
                user_insider = True if decision.lower() == 'yes' else False if decision.lower() == 'no' else None
                insider_marks = 50 if (user_insider is not None and user_insider == correct_insider) else 0
                obj_marks = scenario_marks + insider_marks
        except: obj_marks = None

        confidence_marks = (confidence_score - 1) * 25
        confidence_text = f"Rated {confidence_score}/5"
        case_type = "Without Explainability" if phase_num==1 else "With Explainability"

        try:
            save_response(student_id, user_number, scenario, decision, confidence_text, case_type,
                          obj_marks, confidence_marks, phase_time, total_time)
            st.session_state[f'phase{phase_num}_done'] = True
            st.success("Phase Submitted!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    # --- Questionnaire ---
    if st.session_state.get(f'phase{phase_num}_done', False) and not st.session_state.get(f'questionnaire_p{phase_num}_done', False):
        st.markdown("---")
        st.subheader(f"Questionnaire (Phase {phase_num})")
        st.info("Rate the following (1=Strongly Disagree, 5=Strongly Agree)")
        
        def radio_question(q, key):
            return st.radio(q, [1,2,3,4,5], index=2, key=key, horizontal=True)

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
            q9 = radio_question("9. The ROC curve was useful.", f"q9_p{phase_num}")
        
        q10 = st.text_area("10. Comments?", height=80, key=f"q10_p{phase_num}")

        if st.button("Submit Questionnaire", key=f"submit_q_p{phase_num}"):
            answers = [q1,q2,q3,q4,q5,q6,q7,q8,q9,q10]
            save_questionnaire_phase(student_id, user_number, phase_num, answers)
            
            # Compute Trust
            trust = compute_trust_scores(student_id, user_number, phase_num, ground_truth_df, dict_df)
            try:
                save_trust_score(student_id, user_number, phase_num, trust['objective'], trust['subjective'], trust['combined'])
                st.session_state[f'questionnaire_p{phase_num}_done'] = True
                
                # TRIGGER SCROLL TO TOP ON RERUN
                st.session_state['trigger_scroll'] = True 
                
                st.success("Questionnaire Submitted!")
                st.rerun()
            except Exception as e:
                st.warning(f"Error: {e}")

# -------------------- HELPERS --------------------
def get_last_val(table, col, sid, uid, phase):
    if sid is None or uid is None: return None
    case_type = "Without Explainability" if phase == 1 else "With Explainability"
    conn = sqlite3.connect(DB_RESPONSES)
    try:
        df = pd.read_sql_query(f"SELECT {col} FROM responses WHERE student_id=? AND user_number=? AND case_type=? ORDER BY timestamp DESC LIMIT 1",
                               conn, params=(sid, uid, case_type))
        if not df.empty: return df.iloc[0][0]
    finally: conn.close()
    return None

# ==================== MAIN ====================
def main():
    # TRIGGER SCROLL IF FLAG SET
    if st.session_state.get('trigger_scroll', False):
        scroll_to_top()
        st.session_state['trigger_scroll'] = False

    st.title(APP_TITLE)
    page_choice = st.radio("Navigation:", ["Instructions", "Phase 1", "Phase 2"], index=0, horizontal=True)

    with st.sidebar:
        st.header("Admin")
        if st.button("Initialize DBs"):
            initialize_all_databases()
        with st.expander("📥 Download DBs"):
            for db in DB_FILES.keys():
                if os.path.exists(db):
                    with open(db,"rb") as f: st.download_button(db, f, file_name=db)
        
        st.header("Reports")
        generate_report = st.button("Create Professional PDF Report")

    create_tables_if_missing()
    ground_truth_df = load_ground_truth()
    dict_df = load_dictionary_scenario()

    # -------------------- PROFESSIONAL PDF REPORT --------------------
    if generate_report:
        sid = st.session_state.get('student_id', None)
        u1 = st.session_state.get('phase1_user', None)
        u2 = st.session_state.get('phase2_user', None)
        display_user = u1 if u1 is not None else u2

        # Gather Data
        t1 = compute_trust_scores(sid, u1, 1, ground_truth_df, dict_df) if (sid and u1) else {}
        t2 = compute_trust_scores(sid, u2, 2, ground_truth_df, dict_df) if (sid and u2) else {}
        
        p1_time = get_last_val("responses", "phase_time", sid, u1, 1)
        p2_time = get_last_val("responses", "phase_time", sid, u2, 2)
        p1_obj = get_last_val("responses", "marks", sid, u1, 1)
        p2_obj = get_last_val("responses", "marks", sid, u2, 2)
        p1_conf = get_last_val("responses", "confidence_marks", sid, u1, 1)
        p2_conf = get_last_val("responses", "confidence_marks", sid, u2, 2)

        # Build PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        
        # Custom Styles
        title_style = ParagraphStyle('MainTitle', parent=styles['Heading1'], fontSize=18, alignment=TA_CENTER, spaceAfter=20, textColor=colors.darkblue)
        header_style = ParagraphStyle('Header2', parent=styles['Heading2'], fontSize=14, spaceBefore=15, spaceAfter=10, textColor=colors.darkblue, borderPadding=5)
        normal_style = styles['Normal']
        
        elements = []

        # Title
        elements.append(Paragraph("Insider Threat Assessment Report", title_style))
        elements.append(Paragraph(f"Generated on: {now_toronto().strftime('%Y-%m-%d %H:%M')}", ParagraphStyle('Date', parent=normal_style, alignment=TA_CENTER)))
        elements.append(Spacer(1, 0.3*inch))

        # Participant Info Table
        info_data = [
            ["Participant ID", str(sid) if sid else "N/A"],
            ["Suspect User ID", str(display_user) if display_user else "N/A"],
            ["Assessment Date", now_toronto().strftime("%Y-%m-%d")]
        ]
        info_table = Table(info_data, colWidths=[2*inch, 3*inch], hAlign='LEFT')
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
            ('TEXTCOLOR', (0,0), (0,-1), colors.black),
            ('GRID', (0,0), (-1,-1), 1, colors.white),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 0.4*inch))

        # COMPARATIVE ANALYSIS TABLE
        elements.append(Paragraph("Comparative Analysis: Phase 1 vs Phase 2", header_style))
        
        # Calculate diffs
        def diff_str(v1, v2):
            if v1 is None or v2 is None: return "-"
            d = v2 - v1
            return f"{d:+.2f}"
            
        comp_data = [
            ["Metric", "Phase 1\n(No Explainability)", "Phase 2\n(With Explainability)", "Difference"],
            ["Objective Trust", t1.get('objective', '-'), t2.get('objective', '-'), diff_str(t1.get('objective'), t2.get('objective'))],
            ["Subjective Trust", t1.get('subjective', '-'), t2.get('subjective', '-'), diff_str(t1.get('subjective'), t2.get('subjective'))],
            ["Combined Trust", t1.get('combined', '-'), t2.get('combined', '-'), diff_str(t1.get('combined'), t2.get('combined'))],
            ["Performance Marks", p1_obj if p1_obj is not None else '-', p2_obj if p2_obj is not None else '-', diff_str(p1_obj, p2_obj)],
            ["Confidence (0-100)", p1_conf if p1_conf is not None else '-', p2_conf if p2_conf is not None else '-', diff_str(p1_conf, p2_conf)],
            ["Time Taken (sec)", f"{p1_time:.2f}" if p1_time else '-', f"{p2_time:.2f}" if p2_time else '-', diff_str(p1_time, p2_time)]
        ]

        t = Table(comp_data, colWidths=[2*inch, 1.8*inch, 1.8*inch, 1.2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.aliceblue]),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.4*inch))

        # Interpretations
        elements.append(Paragraph("Interpretations & Executive Summary", header_style))
        
        trust_change = "No Significant Change"
        if t1.get("combined") and t2.get("combined"):
            diff = t2["combined"] - t1["combined"]
            if diff > 0: trust_change = "Trust Increased (Positive Impact of Explainability)"
            elif diff < 0: trust_change = "Trust Decreased (Skepticism Induced)"
        
        summary_text = f"""
        <b>Trust Trend:</b> {trust_change}.<br/><br/>
        <b>Analysis:</b> The introduction of SHAP-based explainability in Phase 2 allows for transparent validation of model anomalies. 
        An increase in trust typically indicates that the explanations reinforced the user's confidence in the AI's logic. 
        A decrease may indicate that the explanations revealed potential flaws or misalignments with analyst intuition.
        """
        elements.append(Paragraph(summary_text, normal_style))

        try:
            doc.build(elements)
            st.sidebar.download_button("Download Report (PDF)", buffer.getvalue(), f"Report_{sid}.pdf", "application/pdf")
            st.success("Report generated successfully (See Sidebar).")
        except Exception as e:
            st.error(f"PDF Generation Error: {e}")

    # --- Routing ---
    if page_choice=="Instructions":
        st.markdown("## Instructions\n1. Enter Participant ID & User ID.\n2. Review Evidence & Submit.\n3. Complete Questionnaire.")
    elif page_choice=="Phase 1": phase_ui_common(1, ground_truth_df, dict_df)
    elif page_choice=="Phase 2": phase_ui_common(2, ground_truth_df, dict_df)

if __name__=="__main__":
    main()
