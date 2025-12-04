# save as insider_study_app.py
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import io
import os
from matplotlib.backends.backend_pdf import PdfPages
import glob
import seaborn as sns

# --- NEW IMPORTS FOR PDF GEN ---
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# -------------------- CONFIG / UTILITIES --------------------
APP_TITLE = "Insider Threat Study"

TORONTO_TZ = ZoneInfo("America/Toronto")

# DB filenames (used through the app)
DB_RESPONSES = "student_responses.db"
DB_FEEDBACK = "feedback.db"
DB_TRUST = "trust.db"
DB_QUESTIONNAIRE = "questionnaire.db"  # kept for completeness
GROUND_TRUTH_CSV = "ground_truth.csv"   # expected in app folder
DICTIONARY_SCENARIO_CSV = "dictionary_scenario.csv"  # uploaded file path

def now_toronto():
    return datetime.now(TORONTO_TZ)

def format_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

# -------------------- DB SCHEMAS --------------------
DB_FILES = {
    "feedback.db": '''
    CREATE TABLE IF NOT EXISTS questionnaire_phase (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        user_number INTEGER,
        phase INTEGER,
        q1 INTEGER,
        q2 INTEGER,
        q3 INTEGER,
        q4 INTEGER,
        q5 INTEGER,
        q6 INTEGER,
        q7 INTEGER,
        q8 INTEGER,
        q9 INTEGER,
        q10 TEXT,
        timestamp TEXT
    );
    ''',
    "questionnaire.db": '''
    CREATE TABLE IF NOT EXISTS analyst_info (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        user_number INTEGER,
        name TEXT,
        email TEXT,
        timestamp TEXT
    );
    ''',
    "trust.db": '''
    CREATE TABLE IF NOT EXISTS trust_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        user_number INTEGER,
        phase INTEGER,
        objective_trust REAL,
        subjective_trust REAL,
        combined_trust REAL,
        timestamp TEXT
    );
    ''',
    # <-- UPDATED schema: added confidence_marks column -->
    "student_responses.db": '''
    CREATE TABLE IF NOT EXISTS responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        user_number INTEGER,
        scenario INTEGER,
        answer TEXT,
        evidence TEXT,
        case_type TEXT,
        marks INTEGER,                -- objective marks (scenario + insider) 0..100 or NULL
        confidence_marks INTEGER,     -- confidence-based marks (I'm Sure=100, I'm Not Sure=50, I Don't Know=0)
        phase_time REAL,
        total_time REAL,
        timestamp TEXT
    );
    '''
}

# -------------------- SESSION STATE FLAGS (safe init) --------------------
if 'db_initialized' not in st.session_state:
    st.session_state['db_initialized'] = False

for db_name in DB_FILES.keys():
    flag = db_name.replace('.', '_') + "_submitted"
    if flag not in st.session_state:
        st.session_state[flag] = False

# -------------------- DB HELPER FUNCTIONS --------------------
def _init_single_db_file(db_path, schema_sql):
    """
    Recreate (drop existing tables) and create the schema fresh.
    Use with caution: this will erase existing data in that DB.
    """
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
        flag = db_name.replace('.', '_') + "_submitted"
        st.session_state[flag] = False
    st.session_state['db_initialized'] = True
    st.success("Databases re-initialized (tables recreated).")

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
    """
    Saves a response. Now stores both objective_marks (marks) and confidence_marks (new column).
    objective_marks can be None (if dictionary missing) — saved as NULL in DB.
    confidence_marks is integer (0/50/100).
    """
    toronto_time = now_toronto().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_RESPONSES)
    c = conn.cursor()
    # convert None objective_marks to NULL by passing None
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
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data
def load_dictionary_scenario(path=DICTIONARY_SCENARIO_CSV):
    """
    Expect columns: user_number, user_name, scenario
    user_number should be integer (or convertible).
    scenario typically integer (0 = normal / non-insider; >0 insider types).
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=['user_number','user_name','scenario'])
    df = pd.read_csv(path, sep=None, engine='python')  # auto-detect sep
    df.columns = [c.strip() for c in df.columns]
    # ensure correct dtypes
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
    if ground_truth_df is None or ground_truth_df.empty:
        return None
    possible_user_cols = ['user', 'user_number', 'user_id', 'userid', 'user number', 'id']
    user_col = next((c for c in ground_truth_df.columns if c.strip().lower() in possible_user_cols), ground_truth_df.columns[0])
    row = ground_truth_df[ground_truth_df[user_col] == user_number]
    if row.empty:
        return None
    row = row.iloc[0]
    mapping = {}
    for c in ground_truth_df.columns:
        cl = c.strip().lower()
        if cl == 'tp' or 'true positive' in cl: mapping['TP'] = c
        if cl == 'tn' or 'true negative' in cl: mapping['TN'] = c
        if cl == 'fp' or 'false positive' in cl: mapping['FP'] = c
        if cl == 'fn' or 'false negative' in cl: mapping['FN'] = c

    for key in ['TP','TN','FP','FN']:
        if key not in mapping:
            for c in ground_truth_df.columns:
                if c.strip().lower() == key.lower():
                    mapping[key] = c
                    break
    tp = int(row[mapping['TP']]) if 'TP' in mapping and pd.notna(row[mapping['TP']]) else 0
    tn = int(row[mapping['TN']]) if 'TN' in mapping and pd.notna(row[mapping['TN']]) else 0
    fp = int(row[mapping['FP']]) if 'FP' in mapping and pd.notna(row[mapping['FP']]) else 0
    fn = int(row[mapping['FN']]) if 'FN' in mapping and pd.notna(row[mapping['FN']]) else 0

    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}

def compute_trust_scores(student_id, user_number, phase, ground_truth_df=None, dict_df=None):
    """
    Compute subjective, objective and combined trust using the dictionary_scenario:
      - Objective: scenario correctness (50) + insider decision correctness (50) -> 0..100
      - Subjective: mean(q1..q9) * 20, if available.
      - Combined: subjective + objective
    This function reads the last saved response for the (student_id, user_number, phase).
    """
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

    # fetch last response for this phase (case_type mapping)
    case_label = {1: "Without Explainability", 2: "With Explainability"}.get(phase)
    user_response = None
    if case_label is not None:
        conn = sqlite3.connect(DB_RESPONSES)
        rdf = pd.read_sql_query(
            "SELECT scenario, answer FROM responses WHERE student_id=? AND user_number=? AND case_type=? ORDER BY timestamp DESC LIMIT 1",
            conn, params=(student_id, user_number, case_label)
        )
        conn.close()
        if not rdf.empty:
            user_response = rdf.iloc[0].to_dict()

    # dictionary lookup
    if dict_df is None:
        dict_df = load_dictionary_scenario()

    objective_score = None
    if user_response is not None and dict_df is not None and not dict_df.empty:
        try:
            correct_row = dict_df[dict_df['user_number'] == int(user_number)]
            if correct_row.empty:
                # If user_number not found in dictionary, we cannot grade scenario; leave objective as None
                objective_score = None
            else:
                correct_scenario = int(correct_row.iloc[0]['scenario'])
                # compute scenario marks (50)
                submitted_scenario = int(user_response.get('scenario', -1))
                scenario_marks = 50 if submitted_scenario == correct_scenario else 0

                # compute insider marks (50): correct insider decision means:
                correct_insider = (correct_scenario != 0)
                user_answer = str(user_response.get('answer', '')).strip().lower()
                user_insider = True if user_answer == 'yes' else False if user_answer == 'no' else None
                insider_marks = 0
                if user_insider is not None:
                    insider_marks = 50 if (user_insider == correct_insider) else 0

                objective_score = round(float(scenario_marks + insider_marks),2)
        except Exception:
            objective_score = None

    # Combined per instruction
    combined = None
    if subj is not None and objective_score is not None:
        combined = round(float(subj + objective_score),2)
    elif subj is None:
        combined = objective_score
    elif objective_score is None:
        combined = subj

    return {"subjective": subj, "objective": objective_score, "combined": combined}

# ==================== PHASE UI ====================
def phase_ui_common(phase_num, ground_truth_df, dict_df):
    st.header(f"Phase {phase_num} — {'Without Explainability' if phase_num==1 else 'With Explainability'}")
    default_student = st.session_state.get('student_id',1)
    student_id = st.number_input("SOC Analyst (Student ID)", min_value=1, max_value=9999, value=int(default_student), key=f"sid_p{phase_num}")
    st.session_state['student_id'] = student_id

    user_number = st.number_input("Suspected Insider (User Number)", min_value=0, max_value=9999, value=0, key=f"user_p{phase_num}")
    st.session_state[f'phase{phase_num}_user'] = user_number

    # --- new: show username in a disabled (grayed-out) box ---
    username_display = ""
    if dict_df is not None and not dict_df.empty:
        matched = dict_df[dict_df['user_number'] == int(user_number)]
        if not matched.empty:
            username_display = str(matched.iloc[0].get('user_name',''))
    st.text_input("User Name (from dictionary)", value=username_display, disabled=True, key=f"username_p{phase_num}")

    # Ensure conf exists to avoid UnboundLocalError later
    conf = None

    if not st.session_state.get(f'phase{phase_num}_done', False):
        st.session_state[f'p{phase_num}_start'] = time.time()
        st.session_state['total_start'] = st.session_state.get('total_start', time.time())
    else:
        saved_phase_time = st.session_state.get(f'phase{phase_num}_time', None)
        if saved_phase_time is not None:
            st.info(f"Phase completed. Recorded Phase time: {format_time(saved_phase_time)}")

    tab_loss, tab_roc, tab_conf, tab_cdf, tab_raw = st.tabs(
        ["Loss", "ROC", "Confusion Matrix", "CDF Curve", "Raw Data"])

    with tab_loss:
        loss_path = f"./user_{user_number:03d}/user_{user_number:03d}_anomaly_scores.png"
        if os.path.exists(loss_path):
            st.image(loss_path, use_container_width=True)
        else:
            st.info(f"Place {loss_path} in app directory for Loss figure.")

    with tab_roc:
        roc_path = f"./user_{user_number:03d}/user_{user_number:03d}_roc_curve.png"
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
        else:
            st.info(f"Place {roc_path} in app directory for ROC figure.")

    with tab_conf:
        if not ground_truth_df.empty:
            conf_temp = _extract_confusion_for_user(ground_truth_df, user_number)
            if conf_temp:
                fig, ax = plt.subplots()
                cm = [[conf_temp['TP'], conf_temp['FN']], [conf_temp['FP'], conf_temp['TN']]]
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred Yes","Pred No"], yticklabels=["Actual Yes","Actual No"], ax=ax)
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)
            else:
                st.info("Confusion matrix not available for this user.")
        else:
            st.info("Confusion matrix image not available.")

    with tab_cdf:
        scores_path_png = f"./user_{user_number:03d}/user_{user_number:03d}_cdf_curve.png"
        if os.path.exists(scores_path_png):
            st.image(scores_path_png, use_container_width=True)
        else:
            st.info("CDF curve not available.")
    
    with tab_raw:
        log_file_path = 'Output2000.log'
        try:
            with open(log_file_path, 'r') as f:
                log_content = f.read()
        except FileNotFoundError:
            st.error(f"Log file '{log_file_path}' not found.")
        except Exception as e:
            st.error(f"Error reading log file: {e}")
        else:
            testing_index = log_content.find("Testing...")
            if testing_index == -1:
                st.warning("'Testing...' section not found in the log file.")
            else:
                log_content = log_content[testing_index:]
                import re
                pattern = re.compile(r'\[(\d+)\] DAY (\d{4}-\d{2}-\d{2}) : ([\d.]+) => \(ans,pred\) (True|False):(True|False)\]')
                entries = []
                for line in log_content.splitlines():
                    match = pattern.match(line)
                    if match:
                        entries.append({
                            'User': int(match.group(1)),
                            'Date': match.group(2),
                            'Loss': float(match.group(3)),
                            'Ans': match.group(4) == 'True',
                            'Pred': match.group(5) == 'True',
                            'Match': match.group(4) == match.group(5)
                        })
                if not entries:
                    st.info("No entries found in the log file.")
                else:
                    df = pd.DataFrame(entries)
                    user_data = df[df['User'] == user_number].reset_index(drop=True)
                    st.dataframe(user_data)

    # Phase 2 SHAP images
    if phase_num == 2:
        tab1, tab2 = st.tabs(["Beeswarm", "Waterfall"])
        with tab1:
            path = f"./user_{user_number:03d}/user_{user_number:03d}_shap_beeswarm_test.png"
            if os.path.exists(path):
                st.image(path, width=300)
            else:
                st.info("Beeswarm plot not available")
        with tab2:
            pattern = f"./user_{user_number:03d}/user_{user_number:03d}_shap_waterfall_*.png"
            files = sorted(glob.glob(pattern))
            if files:
                st.markdown("### Waterfall Plots")
                for f in files:
                    st.image(f, width=300)
            else:
                st.info("No Waterfall plots found for this user.")

    # ------------------- Decision & Scenario -------------------
    decision = st.radio("Insider?", ["No","Yes"], index=0, key=f"p{phase_num}_decision")
    confidence = st.radio("Confidence", ["I'm Sure","I'm Not Sure","I Don't Know"], index=0, key=f"p{phase_num}_confidence")
    
    if decision=="No":
        scenario=-1
        st.selectbox("Scenario", [-1], format_func=scenario_label, index=0, disabled=True, key=f"p{phase_num}_scenario_no")
    else:
        scenario=st.selectbox("Scenario?", SCENARIO_OPTIONS[1:5]+[-1], format_func=scenario_label, index=0, key=f"p{phase_num}_scenario_yes")

    # ------------------- Submit Phase -------------------
    if st.button(f"Submit Phase {phase_num}", key=f"submit_p{phase_num}") and not st.session_state.get(f'phase{phase_num}_done', False):
        phase_time = time.time() - st.session_state.get(f'p{phase_num}_start', time.time())
        total_time = time.time() - st.session_state.get('total_start', time.time())
        st.session_state[f'phase{phase_num}_time'] = phase_time
        st.session_state[f'phase{phase_num}_total_time'] = total_time

        # compute objective marks using dictionary (scenario correctness (50) + insider correctness (50))
        obj_marks = None
        try:
            correct_row = dict_df[dict_df['user_number'] == int(user_number)]
            if not correct_row.empty:
                correct_scenario = int(correct_row.iloc[0]['scenario'])
                scenario_marks = 50 if int(scenario) == correct_scenario else 0
                correct_insider = (correct_scenario != 0)
                user_insider = True if decision.strip().lower() == 'yes' else False if decision.strip().lower() == 'no' else None
                insider_marks = 50 if (user_insider is not None and user_insider == correct_insider) else 0
                obj_marks = scenario_marks + insider_marks
            else:
                obj_marks = None
        except Exception as e:
            obj_marks = None

        # -------------------- Confidence → Marks Mapping --------------------
        confidence_marks_map = {
            "I'm Sure": 100,
            "I'm Not Sure": 50,
            "I Don't Know": 0
        }
        confidence_marks = confidence_marks_map.get(confidence, 0)

        case_type = "Without Explainability" if phase_num==1 else "With Explainability"

        # save_response now accepts both objective_marks and confidence_marks (Option C)
        try:
            save_response(student_id, user_number, scenario, decision, confidence, case_type,
                          obj_marks, confidence_marks, phase_time, total_time)
            st.session_state[f'phase{phase_num}_done'] = True
            st.success(f"Your response for Phase {phase_num} has been recorded successfully! Objective marks: {obj_marks if obj_marks is not None else 'N/A'}, Confidence marks: {confidence_marks}")
        except Exception as e:
            st.error(f"Failed to save response: {e}")

    # ------------------- Show Questionnaire (ONLY ONCE) -------------------
    if st.session_state.get(f'phase{phase_num}_done', False) and not st.session_state.get(f'questionnaire_p{phase_num}_done', False):
        st.markdown("---")
        st.subheader(f"Questionnaire (Phase {phase_num})")
        st.info("Use radio buttons (1=Strongly Disagree, 5=Strongly Agree)")

        def radio_question(q, key, default=1):
            options = [1, 2, 3, 4, 5]
            return st.radio(q, options, index=default-1, key=key, horizontal=True)

        q1 = radio_question("1. The system was easy to use.", f"q1_p{phase_num}")
        q2 = radio_question("2. I trusted the AI's prediction.", f"q2_p{phase_num}")
        q3 = radio_question("3. The explanations helped me make a better decision.", f"q3_p{phase_num}")
        q4 = radio_question("4. I would use this system in real life.", f"q4_p{phase_num}")
        q5 = radio_question("5. The visualizations were clear.", f"q5_p{phase_num}")
        q6 = radio_question("6. I felt confident in my final answer.", f"q6_p{phase_num}")
        q7 = radio_question("7. The timer helped me focus.", f"q7_p{phase_num}")
        q8 = radio_question("8. I understood the loss values.", f"q8_p{phase_num}")
        q9 = radio_question("9. The ROC curve was useful.", f"q9_p{phase_num}")
        q10 = st.text_area("10. Any other comments?", height=120, key=f"q10_p{phase_num}")

        if st.button(f"Submit questionnaire (Phase {phase_num})", key=f"submit_q_p{phase_num}"):
            answers = [q1,q2,q3,q4,q5,q6,q7,q8,q9,q10]
            save_questionnaire_phase(student_id, user_number, phase_num, answers)

            # Use centralized compute_trust_scores to get objective/subjective/combined
            trust = compute_trust_scores(student_id, user_number, phase_num, ground_truth_df, dict_df)
            obj = trust.get('objective', None)
            subj = trust.get('subjective', None)
            combined = trust.get('combined', None)

            # save trust scores in trust.db
            try:
                save_trust_score(student_id, user_number, phase_num, obj, subj, combined)
                st.success("Trust scores saved to trust.db")
            except Exception as e:
                st.warning(f"Failed to save trust score to DB: {e}")

            st.info(f"Phase {phase_num} Trust Scores:\n"
                    f"Objective Trust: {obj}\n"
                    f"Subjective Trust: {subj}\n"
                    f"Combined: {combined}")

            st.session_state[f'questionnaire_p{phase_num}_done'] = True

# -------------------- Helper: retrieve last confidence/objective marks --------------------
def get_last_phase_time(student_id_local, user_number_local, phase):
    if student_id_local is None or user_number_local is None:
        return None
    case_type_local = "Without Explainability" if phase == 1 else "With Explainability"
    conn = sqlite3.connect(DB_RESPONSES)
    try:
        df = pd.read_sql_query(
            "SELECT phase_time FROM responses WHERE student_id=? AND user_number=? AND case_type=? ORDER BY timestamp DESC LIMIT 1",
            conn,
            params=(student_id_local, user_number_local, case_type_local)
        )
    finally:
        conn.close()
    if not df.empty:
        return float(df['phase_time'].iloc[0])
    return None

def get_last_objective_marks(student_id_local, user_number_local, phase):
    if student_id_local is None or user_number_local is None:
        return None
    case_type_local = "Without Explainability" if phase == 1 else "With Explainability"
    conn = sqlite3.connect(DB_RESPONSES)
    try:
        df = pd.read_sql_query(
            "SELECT marks FROM responses WHERE student_id=? AND user_number=? AND case_type=? ORDER BY timestamp DESC LIMIT 1",
            conn,
            params=(student_id_local, user_number_local, case_type_local)
        )
    finally:
        conn.close()
    if not df.empty:
        val = df['marks'].iloc[0]
        return int(val) if pd.notna(val) else None
    return None

def get_last_confidence_marks(student_id_local, user_number_local, phase):
    if student_id_local is None or user_number_local is None:
        return None
    case_type_local = "Without Explainability" if phase == 1 else "With Explainability"
    conn = sqlite3.connect(DB_RESPONSES)
    try:
        df = pd.read_sql_query(
            "SELECT confidence_marks FROM responses WHERE student_id=? AND user_number=? AND case_type=? ORDER BY timestamp DESC LIMIT 1",
            conn,
            params=(student_id_local, user_number_local, case_type_local)
        )
    finally:
        conn.close()
    if not df.empty:
        val = df['confidence_marks'].iloc[0]
        return int(val) if pd.notna(val) else None
    return None

# ==================== MAIN ====================
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🛡️", initial_sidebar_state="collapsed")
    st.title(APP_TITLE)

    page_choice = st.radio("Navigation:", ["Instructions", "Phase 1", "Phase 2"], index=0, horizontal=True)

    with st.sidebar:
        st.header("Setup")

        if st.button("Initialize Databases"):
            try:
                initialize_all_databases()
            except Exception as e:
                st.error(f"Failed to initialize databases: {e}")

        with st.expander("📥 Download Databases", expanded=True):
            st.write("Download local DB files (enabled until you click Initialize; after Initialize they re-enable per-DB upon first submission).")
            for db_path in DB_FILES.keys():
                submitted_flag = db_path.replace('.', '_') + "_submitted"
                disabled_state = False
                if st.session_state.get('db_initialized', False):
                    disabled_state = not bool(st.session_state.get(submitted_flag, False))
                else:
                    disabled_state = False

                if os.path.exists(db_path):
                    try:
                        with open(db_path, "rb") as f:
                            st.download_button(
                                label=f"Download {db_path}",
                                data=f,
                                file_name=db_path,
                                mime="application/octet-stream",
                                key=f"dl_{db_path}",
                                disabled=disabled_state
                            )
                    except Exception as e:
                        st.warning(f"Could not prepare {db_path} for download: {e}")
                else:
                    st.warning(f"{db_path} not found.")

        st.header("Reports")
        generate_report = st.button("Create PDF Report")

    create_tables_if_missing()
    ground_truth_df = load_ground_truth()
    dict_df = load_dictionary_scenario()

    # -------------------- PDF GENERATION --------------------
    if 'generate_report' in locals() and generate_report:
        student_id = st.session_state.get('student_id', None)
        user_number_p1 = st.session_state.get('phase1_user', None)
        user_number_p2 = st.session_state.get('phase2_user', None)
        display_user_number = user_number_p1 if user_number_p1 is not None else user_number_p2

        t1 = compute_trust_scores(student_id, user_number_p1, 1, ground_truth_df, dict_df) if (student_id is not None and user_number_p1 is not None) else {"objective": None, "subjective": None, "combined": None}
        t2 = compute_trust_scores(student_id, user_number_p2, 2, ground_truth_df, dict_df) if (student_id is not None and user_number_p2 is not None) else {"objective": None, "subjective": None, "combined": None}

        phase1_time = get_last_phase_time(student_id, user_number_p1, 1)
        phase2_time = get_last_phase_time(student_id, user_number_p2, 2)

        # fetch objective & confidence marks for both phases (last saved)
        phase1_obj_marks = get_last_objective_marks(student_id, user_number_p1, 1)
        phase1_conf_marks = get_last_confidence_marks(student_id, user_number_p1, 1)
        phase2_obj_marks = get_last_objective_marks(student_id, user_number_p2, 2)
        phase2_conf_marks = get_last_confidence_marks(student_id, user_number_p2, 2)

        trust_diff = None
        trust_direction = "No Change"
        if t1.get("combined") is not None and t2.get("combined") is not None:
            trust_diff = t2["combined"] - t1["combined"]
            if trust_diff > 0:
                trust_direction = "Trust Increased"
            elif trust_diff < 0:
                trust_direction = "Trust Decreased"

        time_diff = None
        time_direction = "No Change"
        if (phase1_time is not None) and (phase2_time is not None):
            time_diff = phase2_time - phase1_time
            if trust_diff is not None and trust_diff > 0:
                time_direction = "Time Increased"
            elif trust_diff is not None and trust_diff < 0:
                time_direction = "Time Decreased"

        pdf_buffer = io.BytesIO()
        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter,
                                rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)

        elements = []
        elements.append(Paragraph(f"<b>Insider Threat Study Report</b>", styles["Title"]))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(f"SOC Analyst (Student ID): <b>{student_id}</b>", styles["Normal"]))
        elements.append(Paragraph(f"Suspected Insider (User Number): <b>{display_user_number}</b>", styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

        elements.append(Paragraph("<b>Phase 1 — Without Explainability</b>", styles["Heading2"]))
        elements.append(Paragraph(f"Objective Trust: {t1.get('objective')}", styles["Normal"]))
        elements.append(Paragraph(f"Subjective Trust: {t1.get('subjective')}", styles["Normal"]))
        elements.append(Paragraph(f"Combined Trust: {t1.get('combined')}", styles["Normal"]))
        elements.append(Paragraph(f"Objective Marks (scenario+insider): {phase1_obj_marks}", styles["Normal"]))
        elements.append(Paragraph(f"Confidence Marks: {phase1_conf_marks}", styles["Normal"]))
        if phase1_time is not None:
            elements.append(Paragraph(f"Time Taken: {phase1_time:.2f} seconds", styles["Normal"]))
        elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph("<b>Phase 2 — With Explainability</b>", styles["Heading2"]))
        elements.append(Paragraph(f"Objective Trust: {t2.get('objective')}", styles["Normal"]))
        elements.append(Paragraph(f"Subjective Trust: {t2.get('subjective')}", styles["Normal"]))
        elements.append(Paragraph(f"Combined Trust: {t2.get('combined')}", styles["Normal"]))
        elements.append(Paragraph(f"Objective Marks (scenario+insider): {phase2_obj_marks}", styles["Normal"]))
        elements.append(Paragraph(f"Confidence Marks: {phase2_conf_marks}", styles["Normal"]))
        if phase2_time is not None:
            elements.append(Paragraph(f"Time Taken: {phase2_time:.2f} seconds", styles["Normal"]))
        elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph("<b>Comparison & Differences</b>", styles["Heading2"]))
        if trust_diff is not None:
            elements.append(Paragraph(f"Combined Trust Difference (Phase 2 - Phase 1): {trust_diff:.2f}", styles["Normal"]))
            elements.append(Paragraph(f"Interpretation: <b>{trust_direction}</b>", styles["Normal"]))
        else:
            elements.append(Paragraph("Trust difference unavailable.", styles["Normal"]))

        if time_diff is not None:
            elements.append(Paragraph(f"Time Difference (Phase 2 - Phase 1): {time_diff:.2f} seconds", styles["Normal"]))
            elements.append(Paragraph(f"Interpretation: <b>{time_direction}</b>", styles["Normal"]))
        else:
            elements.append(Paragraph("Time difference unavailable.", styles["Normal"]))

        elements.append(Spacer(1, 0.2 * inch))

        if trust_diff is not None:
            if trust_diff > 0:
                trust_justification = (
                    "The increase in trust during Phase 2 is expected. The introduction of SHAP-based "
                    "explainability provides participants with clear evidence regarding why the model "
                    "flagged certain events as anomalous. Increased transparency helps participants align "
                    "the model's reasoning with their own understanding of insider threat behavior."
                )
            elif trust_diff < 0:
                trust_justification = (
                    "The decrease in trust during Phase 2 can occur when explanations reveal model weaknesses "
                    "or unexpected feature attributions that differ from analyst expectations."
                )
            else:
                trust_justification = (
                    "Trust remained stable across both phases. This suggests that the explainability "
                    "information neither contradicted nor reinforced participants’ expectations strongly "
                    "enough to change their assessment of the model's reliability."
                )

            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph("<b>Justification for Trust Change</b>", styles["Heading2"]))
            elements.append(Paragraph(trust_justification, styles["Normal"]))

        if time_diff is not None and time_diff > 0:
            justification_text = (
                "The increase in time during Phase 2 likely reflects the cognitive cost of "
                "interpreting model explanations in addition to making the binary decision."
            )
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph("<b>Justification for Increased Time in Phase 2</b>", styles["Heading2"]))
            elements.append(Paragraph(justification_text, styles["Normal"]))

        doc.build(elements)

        st.sidebar.download_button(
            label="Download PDF Report",
            data=pdf_buffer.getvalue(),
            file_name=f"report_student_{student_id}_user_{display_user_number}.pdf",
            mime="application/pdf"
        )

        st.success("PDF report prepared — use the sidebar button to download.")

    if page_choice=="Instructions":
        st.markdown("""
        # Insider Threat Study Instructions
        ### Procedure
        1. Select **Phase 1** or **Phase 2** from the navigation above.
        2. Enter **Your Student ID (acting as SOC analyst)** and **User Number to be checked**.
        3. The phase timer starts automatically when you open the phase (it records time-on-task).
        4. Review the Confusion Matrix, Anomaly Score and ROC images for the user, then make the decision **Insider? (Yes/No)** and select scenario if Yes.
        5. Click **Submit Phase** to stop the timer — after submission the **Questionnaire** will appear below (no timer).
        6. Complete the questionnaire and submit. Trust scores are computed when both phase and questionnaire are saved.

        ### Note
        - Objective trust is now computed as: Scenario correctness (50) + Insider decision correctness (50).
        - Subjective trust is the mean of questionnaire sliders (scaled to 0–100).
        - Combined trust = subjective + objective.
        - **Database change:** responses table now contains `confidence_marks` (0/50/100) in addition to `marks` (objective).
        """)
    elif page_choice=="Phase 1":
        phase_ui_common(1, ground_truth_df, dict_df)
    elif page_choice=="Phase 2":
        phase_ui_common(2, ground_truth_df, dict_df)

if __name__=="__main__":
    main()
