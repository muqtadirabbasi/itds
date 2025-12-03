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

# -------------------- CONFIG / UTILITIES --------------------
APP_TITLE = "Insider Threat Study"
TORONTO_TZ = ZoneInfo("America/Toronto")

# DB filenames (used through the app)
DB_RESPONSES = "student_responses.db"
DB_FEEDBACK = "feedback.db"
DB_TRUST = "trust.db"
DB_QUESTIONNAIRE = "questionnaire.db"  # kept for completeness
GROUND_TRUTH_CSV = "ground_truth.csv"   # expected in app folder

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
    "student_responses.db": '''
    CREATE TABLE IF NOT EXISTS responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        user_number INTEGER,
        scenario INTEGER,
        answer TEXT,
        evidence TEXT,
        case_type TEXT,
        marks INTEGER,
        phase_time REAL,
        total_time REAL,
        timestamp TEXT
    );
    '''
}

# -------------------- SESSION STATE FLAGS (safe init) --------------------
if 'db_initialized' not in st.session_state:
    # db_initialized controls whether "Initialize Databases" has been pressed.
    # When False -> download buttons are enabled.
    # When True  -> download buttons are disabled until a submission re-enables them per-db.
    st.session_state['db_initialized'] = False

# Per-DB "first-submission" flags. False means no submission yet.
for db_name in DB_FILES.keys():
    flag = db_name.replace('.', '_') + "_submitted"
    if flag not in st.session_state:
        st.session_state[flag] = False

# -------------------- DB HELPER FUNCTIONS --------------------
def _init_single_db_file(db_path, schema_sql):
    """Drop existing tables (except sqlite_sequence) and recreate schema for that DB file."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # drop all tables except sqlite_sequence
    tables = c.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    for (t,) in tables:
        if t == "sqlite_sequence":
            continue
        c.execute(f"DROP TABLE IF EXISTS {t};")
    # create new schema
    c.executescript(schema_sql)
    conn.commit()
    conn.close()

def initialize_all_databases():
    """Initialize (drop & recreate) all DB files defined in DB_FILES."""
    for db_path, schema in DB_FILES.items():
        _init_single_db_file(db_path, schema)
    # After re-creating DBs, none should be considered "submitted"
    for db_name in DB_FILES.keys():
        flag = db_name.replace('.', '_') + "_submitted"
        st.session_state[flag] = False
    # mark as initialized so buttons become disabled until submissions
    st.session_state['db_initialized'] = True

def create_tables_if_missing():
    """Create any missing tables (non-destructive) at startup."""
    for db_path, schema in DB_FILES.items():
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            conn.executescript(schema)
            conn.commit()
        finally:
            conn.close()

# -------------------- PERSISTENCE HELPERS (also set submitted flags) --------------------
def save_response(student_id, user_number, scenario, answer, confidence, case_type, marks, phase_time, total_time):
    toronto_time = now_toronto().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_RESPONSES)
    c = conn.cursor()
    c.execute(
        """INSERT INTO responses (student_id,user_number,scenario,answer,evidence,case_type,marks,phase_time,total_time,timestamp)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (int(student_id), int(user_number), int(scenario), answer, confidence, case_type, int(marks), float(phase_time), float(total_time), toronto_time)
    )
    conn.commit()
    conn.close()
    # mark that student_responses.db has a submission so its download can be re-enabled
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
    # mark that feedback.db has a submission so its download can be re-enabled
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
    # mark that trust.db has a submission so its download can be re-enabled
    st.session_state['trust_db_submitted'] = True

# -------------------- DATA LOADERS --------------------
@st.cache_data
def load_ground_truth(path=GROUND_TRUTH_CSV):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
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
    return SCENARIO_LABELS.get(x, str(x))

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

    # fallback find by exact label
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

def compute_trust_scores(student_id, user_number, phase, ground_truth_df=None):
    """
    Compute subjective, objective and combined trust:
      - Objective: 100 if (decision == Yes and TP>0) OR (decision == No and TP==0), else 0.
      - Subjective: mean(q1..q9) * 20, if available.
      - Combined: subjective + objective (handles missing pieces).
    """
    subj = None
    obj = None

    # subjective: read questionnaire entries (phase specific)
    conn = sqlite3.connect(DB_FEEDBACK)
    qdf = pd.read_sql_query(
        "SELECT q1,q2,q3,q4,q5,q6,q7,q8,q9 FROM questionnaire_phase WHERE student_id=? AND user_number=? AND phase=?",
        conn, params=(student_id, user_number, phase)
    )
    conn.close()
    if not qdf.empty:
        vals = qdf.mean(axis=0).values.astype(float)
        subj = float(vals.mean() * 20.0)

    # objective: last decision from responses DB, filtered by case_type mapping
    case_label = {1: "Without Explainability", 2: "With Explainability"}.get(phase)
    user_decision = None
    if case_label is not None:
        conn = sqlite3.connect(DB_RESPONSES)
        rdf = pd.read_sql_query(
            "SELECT answer FROM responses WHERE student_id=? AND user_number=? AND case_type=? ORDER BY timestamp DESC LIMIT 1",
            conn, params=(student_id, user_number, case_label)
        )
        conn.close()
        if not rdf.empty:
            user_decision = str(rdf['answer'].iloc[0])

    # ground truth conf (TP only used per Option A)
    conf = _extract_confusion_for_user(ground_truth_df, user_number) if ground_truth_df is not None else None
    tp = int(conf.get('TP', 0)) if conf else 0

    if user_decision is not None:
        ud = user_decision.strip().lower()
        if (ud == "yes" and tp > 0) or (ud == "no" and tp == 0):
            obj = 100.0
        else:
            obj = 0.0

    # Combined per instruction: subjective + objective
    combined = None
    if subj is not None and obj is not None:
        combined = subj + obj
    elif subj is None:
        combined = obj
    elif obj is None:
        combined = subj

    return {"subjective": subj, "objective": obj, "combined": combined}

# ==================== PHASE UI ====================
def phase_ui_common(phase_num, ground_truth_df):
    st.header(f"Phase {phase_num} — {'Without Explainability' if phase_num==1 else 'With Explainability'}")
    default_student = st.session_state.get('student_id',1)
    student_id = st.number_input("Student ID", min_value=1, max_value=9999, value=int(default_student), key=f"sid_p{phase_num}")
    st.session_state['student_id'] = student_id
    user_number = st.number_input("User Number", min_value=0, max_value=9999, value=0, key=f"user_p{phase_num}")
    st.session_state[f'phase{phase_num}_user'] = user_number

    # Ensure conf exists to avoid UnboundLocalError later
    conf = None

    if not st.session_state.get(f'phase{phase_num}_done', False):
        st.session_state[f'p{phase_num}_start'] = time.time()
        st.session_state['total_start'] = time.time()
    else:
        saved_phase_time = st.session_state.get(f'phase{phase_num}_time', None)
        if saved_phase_time is not None:
            st.info(f"Phase completed. Recorded Phase time: {format_time(saved_phase_time)}")

    # ------------------- Loss / ROC / Confusion / CDF tabs -------------------
   #tab_loss, tab_roc, tab_conf, tab_cdf = st.tabs(["Loss","ROC","Confusion Matrix","CDF Curve"])

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
    
    # ==================== RAW DATA TAB ====================
    with tab_raw:  # assuming raw_data = st.tab("Raw Data") defined somewhere in your tab setup
        #st.header("📊 Raw Data Viewer")

        log_file_path = '/work/Output2000.log'

        # Read log file
        try:
            with open(log_file_path, 'r') as f:
                log_content = f.read()
        except FileNotFoundError:
            st.error(f"Log file '{log_file_path}' not found.")
        except Exception as e:
            st.error(f"Error reading log file: {e}")
        else:
            # Extract section after "Testing..."
            testing_index = log_content.find("Testing...")
            if testing_index == -1:
                st.warning("'Testing...' section not found in the log file.")
            else:
                log_content = log_content[testing_index:]

                # Parse log lines
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

                    # Filter by user
                    user_list = sorted(df['User'].unique())
                    #selected_user = st.selectbox("Select User", user_number)

                    user_data = df[df['User'] == user_number].reset_index(drop=True)
                    st.dataframe(user_data)

                    # Show counts
                    #st.write(user_data[['Ans','Pred']].apply(pd.Series.value_counts))

                    # Download button
                    #csv = user_data.to_csv(index=False).encode('utf-8')
                    #st.download_button(
                    #    label="Download CSV",
                    #    data=csv,
                    #    file_name=f'user_{user_number}_raw_data.csv',
                    #    mime='text/csv'
                  #)


    # ------------------- Phase 2 explanation plots -------------------
    if phase_num == 2:
        tab1, tab2 = st.tabs(["Beeswarm", "Waterfall"])
        with tab1:
            path = f"./user_{user_number:03d}/user_{user_number:03d}_shap_beeswarm_test.png"
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.info("Beeswarm plot not available")
        with tab2:
            pattern = f"./user_{user_number:03d}/user_{user_number:03d}_shap_waterfall_*.png"
            files = sorted(glob.glob(pattern))
            if files:
                st.markdown("### Waterfall Plots")
                for f in files:
                    st.image(f, use_container_width=True)
            else:
                st.info("No Waterfall plots found for this user.")

    # ------------------- Decision & Scenario -------------------
    decision = st.radio("Insider?", ["No","Yes"], index=0, key=f"p{phase_num}_decision")
    confidence = st.radio("Confidence", ["I'm Sure","I'm Not Sure","I Don't Know"], index=0, key=f"p{phase_num}_confidence")
    
    if decision=="No":
        scenario=-1
        st.selectbox("Scenario", [-1], format_func=scenario_label, index=0, disabled=True, key=f"p{phase_num}_scenario_no")
    else:
        # Only show insider scenarios + "I Don't Know"
        scenario=st.selectbox("Scenario?", SCENARIO_OPTIONS[1:5]+[-1], format_func=scenario_label, index=0, key=f"p{phase_num}_scenario_yes")

    # ------------------- Submit Phase -------------------
    if st.button(f"Submit Phase {phase_num}", key=f"submit_p{phase_num}") and not st.session_state.get(f'phase{phase_num}_done', False):
        phase_time = time.time() - st.session_state.get(f'p{phase_num}_start', time.time())
        total_time = time.time() - st.session_state.get('total_start', time.time())
        st.session_state[f'phase{phase_num}_time'] = phase_time
        st.session_state[f'phase{phase_num}_total_time'] = total_time

        # set conf now (so questionnaire handler can use it)
        conf = _extract_confusion_for_user(ground_truth_df, user_number) if not ground_truth_df.empty else None

        # For marks calculation we keep existing behavior (uses TP+FN as "actual positives")
        actual_positives = int(conf.get('TP',0)) + int(conf.get('FN',0)) if conf else 0
        is_mal = actual_positives > 0
        marks = 1 if ((decision=="Yes") == is_mal) else 0
        case_type = "Without Explainability" if phase_num==1 else "With Explainability"

        save_response(student_id,user_number,scenario,decision,confidence,case_type,marks,phase_time,total_time)
        st.session_state[f'phase{phase_num}_done'] = True
        st.success(f"Your response for Phase {phase_num} has been recorded successfully!")

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
            trust = compute_trust_scores(student_id, user_number, phase_num, ground_truth_df)
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
                    f"Combined Trust: {combined}")

            st.session_state[f'questionnaire_p{phase_num}_done'] = True

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
                # initialize_all_databases sets st.session_state['db_initialized']=True and clears per-db _submitted flags
                st.success("All databases have been initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize databases: {e}")

        # grouped, collapsible download area
        with st.expander("📥 Download Databases", expanded=True):
            st.write("Download local DB files (enabled until you click Initialize; after Initialize they re-enable per-DB upon first submission).")
            for db_path in DB_FILES.keys():
                # compute whether this db's download should be disabled:
                # disabled if (db_initialized is True) AND (this DB has NOT yet been submitted to)
                submitted_flag = db_path.replace('.', '_') + "_submitted"
                disabled_state = False
                if st.session_state.get('db_initialized', False):
                    # if initialized, downloads should be disabled until corresponding DB receives first submission
                    disabled_state = not bool(st.session_state.get(submitted_flag, False))
                else:
                    # if not initialized, downloads should be enabled
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

    # Ensure tables exist at startup (non-destructive)
    create_tables_if_missing()
    ground_truth_df = load_ground_truth()

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

        ### Image layout (Option A)
        Place your images in `/work/user_{user_number}/` with the filenames:
        - `user_{user_number:03d}_anomaly_scores.png`
        - `user_{user_number:03d}_anomaly_scores.csv` (for CDF)
        - `user_{user_number:03d}_roc_curve.png`
        - `user_{user_number:03d}_shap_beeswarm_test.png` (Phase 2)
        - `user_{user_number:03d}_shap_waterfall_*.png` (Phase 2 waterfall files)

        ### Data & Privacy
        - Responses and questionnaires are stored in local SQLite databases (`student_responses.db`, `feedback.db`, `trust.db`).
        - Export DBs via the sidebar for offline analysis.

        ### Notes for methods / reproducibility
        - Objective trust uses **TP only** (Option A).
        - Subjective trust is the mean of questionnaire sliders (scaled to 0–100).
        - Combined trust = subjective + objective (per instruction).
        """)
    elif page_choice=="Phase 1":
        phase_ui_common(1, ground_truth_df)
    elif page_choice=="Phase 2":
        phase_ui_common(2, ground_truth_df)

if __name__=="__main__":
    main()