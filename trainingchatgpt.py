# save as insider_study_app.py
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import time
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from sklearn.metrics import roc_curve, auc
import io
import os
from matplotlib.backends.backend_pdf import PdfPages

# ==================== CONFIG ====================
APP_TITLE = "Insider Threat Study"
TORONTO_TZ = ZoneInfo("America/Toronto")
DB_RESPONSES = "student_responses.db"
DB_FEEDBACK = "feedback.db"
LOG_PATH = "/work/Output2000.log"
BEESWARM_PATH_TEMPLATE = "/work/beeswarm_plot{user}.png"
WATERFALL_PATH_TEMPLATE = "/work/waterfall_plot{user}.png"
GROUND_TRUTH_CSV = "user_true_positives.csv"

# ==================== UTILITIES ====================
def now_toronto():
    return datetime.now(TORONTO_TZ)

def format_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

# ==================== DATABASES ====================
def init_db(path: str, schema_sql: str):
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()
    c.executescript(schema_sql)
    conn.commit()
    conn.close()

RESPONSES_SCHEMA = '''
CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    user_number INTEGER,
    scenario INTEGER,
    answer TEXT,
    confidence TEXT,
    case_type TEXT,
    marks INTEGER,
    phase_time REAL,
    total_time REAL,
    timestamp TEXT
);
'''

QUESTIONNAIRE_PHASE_SCHEMA = '''
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
'''

def init_dbs():
    # This function preserves your original behavior: DROP and recreate tables.
    # It will clear databases when called (keeps original semantics).
    # Clear RESPONSES database
    conn = sqlite3.connect(DB_RESPONSES, check_same_thread=False)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS responses;")
    conn.commit()
    conn.close()

    # Clear FEEDBACK database
    conn = sqlite3.connect(DB_FEEDBACK, check_same_thread=False)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS questionnaire_phase;")
    conn.commit()
    conn.close()

    # Recreate empty tables
    init_db(DB_RESPONSES, RESPONSES_SCHEMA)
    init_db(DB_FEEDBACK, QUESTIONNAIRE_PHASE_SCHEMA)

def create_tables_if_missing():
    # Safe startup: create tables only if they don't exist (no dropping).
    init_db(DB_RESPONSES, RESPONSES_SCHEMA)
    init_db(DB_FEEDBACK, QUESTIONNAIRE_PHASE_SCHEMA)

def save_response(student_id, user_number, scenario, answer, confidence, case_type, marks, phase_time, total_time):
    toronto_time = now_toronto().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_RESPONSES)
    c = conn.cursor()
    c.execute(
        """INSERT INTO responses (student_id,user_number,scenario,answer,confidence,case_type,marks,phase_time,total_time,timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (int(student_id), int(user_number), int(scenario), answer, confidence, case_type, int(marks), float(phase_time), float(total_time), toronto_time)
    )
    conn.commit()
    conn.close()

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

# ==================== DATA LOADERS ====================
@st.cache_data
def load_ground_truth(path=GROUND_TRUTH_CSV):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_log_file(path=LOG_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        content = f.read()
    start = content.find("Testing...")
    section = content[start:] if start != -1 else content
    pattern = re.compile(r"\[(\d+)\] DAY (\d{4}-\d{2}-\d{2}) : ([\d.]+) => \(ans,pred\) (True|False):(True|False)")
    rows = []
    for line in section.splitlines():
        m = pattern.match(line.strip())
        if m:
            rows.append({
                "User Number": int(m.group(1)),
                "Date": m.group(2),
                "Loss": float(m.group(3)),
                "Ans": m.group(4) == "True",
                "Pred": m.group(5) == "True",
            })
    if not rows:
        return None
    return pd.DataFrame(rows)

def get_last_tp_for_user(user_number, log_df):
    if log_df is None or log_df.empty:
        return 0
    user_data = log_df[log_df['User Number'] == user_number]
    if user_data.empty:
        return 0
    # Count True Positives
    TP = ((user_data['Ans'] == True) & (user_data['Pred'] == True)).sum()
    return int(TP)


# ==================== SCENARIO LABELS ====================
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

# ==================== TRUST SCORE ====================
def compute_trust_scores(student_id, user_number, phase, log_df=None):
    subj = None
    obj = None

    # Subjective trust (same as before)
    conn = sqlite3.connect(DB_FEEDBACK)
    qdf = pd.read_sql_query(
        "SELECT q1,q2,q3,q4,q5,q6,q7,q8,q9 FROM questionnaire_phase WHERE student_id=? AND user_number=? AND phase=?",
        conn, params=(student_id, user_number, phase)
    )
    conn.close()
    if not qdf.empty:
        vals = qdf.mean(axis=0).values.astype(float)
        subj = vals.mean() * 20.0

    # Objective trust: compute TP from log
    last_tp = get_last_tp_for_user(user_number, log_df)

    # Get user decision from responses DB
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

    # Compute objective trust
    if user_decision is not None:
        ud = user_decision.strip().lower()
        if ud == "no" and last_tp == 0:
            obj = 100.0
        elif ud == "yes" and last_tp > 0:
            obj = 100.0
        else:
            obj = 0.0
    else:
        obj = None

    combined = None
    if subj is not None and obj is not None:
        combined = 0.6 * subj + 0.4 * obj
    elif subj is None:
        combined = obj
    elif obj is None:
        combined = subj

    return {"subjective": subj, "objective": obj, "combined": combined}


# ==================== PDF REPORT ====================
def generate_pdf_report_bytes(student_id, user_number, log_df, phase=None):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        ax.text(0.01, 0.98, f"Insider Threat Study Report\n\nStudent: {student_id}\nUser: {user_number}\nPhase: {phase}", va='top', wrap=True)
        pdf.savefig(fig); plt.close(fig)

        if log_df is not None:
            user_data = log_df[log_df['User Number']==user_number]
            if not user_data.empty:
                fig, ax = plt.subplots(figsize=(8,4))
                ax.plot(user_data.index+1, user_data['Loss'], marker='o')
                ax.set_title("Loss over time"); ax.set_xlabel("Seq"); ax.set_ylabel("Loss")
                pdf.savefig(fig); plt.close(fig)
                try:
                    fpr,tpr,_ = roc_curve(user_data['Ans'].astype(int), user_data['Loss'])
                    roc_auc = auc(fpr,tpr)
                    fig, ax = plt.subplots(figsize=(6,6))
                    ax.plot(fpr,tpr); ax.plot([0,1],[0,1],'--')
                    ax.set_title(f"ROC AUC={roc_auc:.3f}")
                    pdf.savefig(fig); plt.close(fig)
                except Exception:
                    pass

        for p in [BEESWARM_PATH_TEMPLATE.format(user=user_number),
                  WATERFALL_PATH_TEMPLATE.format(user=user_number)]:
            if os.path.exists(p):
                fig = plt.figure(figsize=(8.27,11.69))
                img = plt.imread(p)
                plt.imshow(img); plt.axis('off')
                pdf.savefig(fig); plt.close(fig)
    buf.seek(0)
    return buf.read()

# ==================== QUESTIONNAIRE CSV helper ====================
def questionnaire_phase_to_csv_bytes(student_id, user_number, phase):
    conn = sqlite3.connect(DB_FEEDBACK)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM questionnaire_phase WHERE student_id=? AND user_number=? AND phase=? ORDER BY timestamp DESC",
            conn, params=(int(student_id), int(user_number), int(phase))
        )
    except Exception:
        df = pd.DataFrame()
    conn.close()
    if df.empty:
        template = pd.DataFrame([{
            "student_id": student_id,
            "user_number": user_number,
            "phase": phase,
            "q1": "",
            "q2": "",
            "q3": "",
            "q4": "",
            "q5": "",
            "q6": "",
            "q7": "",
            "q8": "",
            "q9": "",
            "q10": "",
            "timestamp": ""
        }])
        return template.to_csv(index=False).encode("utf-8")
    else:
        return df.to_csv(index=False).encode("utf-8")

# ==================== PHASE UI (Option B preserved) ====================
def phase_ui_common(phase_num, log_df, ground_truth_df):
    st.header(f"Phase {phase_num} — {'Without Explainability' if phase_num==1 else 'With Explainability'}")
    default_student = st.session_state.get('student_id',1)
    student_id = st.number_input("Student ID", min_value=1, max_value=9999, value=int(default_student), key=f"sid_p{phase_num}")
    st.session_state['student_id'] = student_id

    user_number = st.number_input("User Number", min_value=0, max_value=9999, value=0, key=f"user_p{phase_num}")
    st.session_state[f'phase{phase_num}_user'] = user_number

    phase_done_flag = st.session_state.get(f'phase{phase_num}_done', False)
    running_flag = st.session_state.get(f'p{phase_num}_start') is not None

    if not phase_done_flag and not running_flag:
        st.session_state[f'p{phase_num}_start'] = time.time()
        st.session_state['total_start'] = time.time()

    if not st.session_state.get(f'phase{phase_num}_done', False):
        elapsed = time.time() - st.session_state[f'p{phase_num}_start']
        st.info(f"Phase time (running): {format_time(elapsed)}")
    else:
        saved_phase_time = st.session_state.get(f'phase{phase_num}_time', None)
        if saved_phase_time is not None:
            st.info(f"Phase completed. Recorded Phase time: {format_time(saved_phase_time)}")
        else:
            st.info("Phase completed.")

    # Tabs for Raw Data / Loss / ROC
    tab_data, tab_loss, tab_roc = st.tabs(["Raw Data","Loss","ROC"])
    if log_df is not None:
        user_data = log_df[log_df['User Number']==user_number]
        if not user_data.empty:
            user_data = user_data.copy().reset_index(drop=True)
            user_data.index += 1
            TP=((user_data['Ans']==True)&(user_data['Pred']==True)).sum()
            FP=((user_data['Ans']==False)&(user_data['Pred']==True)).sum()
            FN=((user_data['Ans']==True)&(user_data['Pred']==False)).sum()
            TN=len(user_data)-TP-FP-FN
            acc=(TP+TN)/len(user_data)

            with tab_data:
                st.dataframe(user_data[['Date','Loss','Ans','Pred']], use_container_width=True)
            with tab_loss:
                fig, ax = plt.subplots(figsize=(8,3))
                ax.plot(user_data.index, user_data['Loss'], marker='o')
                ax.set_title("Loss over time"); ax.set_xlabel("Seq"); ax.set_ylabel("Loss")
                st.pyplot(fig)
            with tab_roc:
                try:
                    fpr,tpr,_=roc_curve(user_data['Ans'].astype(int),user_data['Loss'])
                    roc_auc=auc(fpr,tpr)
                    fig, ax = plt.subplots(figsize=(4,4))
                    ax.plot(fpr,tpr); ax.plot([0,1],[0,1],'--')
                    ax.set_title(f"ROC AUC={roc_auc:.3f}")
                    st.pyplot(fig)
                except Exception:
                    st.info("ROC cannot be computed")
            st.warning(f"TP={TP} | FP={FP} | FN={FN} | Acc={acc:.1%}")
        else:
            st.info("User not in log")
    else:
        st.info("Log not found")

    # Phase 2: explanation images
    if phase_num==2:
        tab1,tab2=st.tabs(["Global (Beeswarm)","Local (Waterfall)"])
        with tab1:
            path=BEESWARM_PATH_TEMPLATE.format(user=user_number)
            if os.path.exists(path):
                st.image(path,use_column_width=True)
            else:
                st.info("Beeswarm plot not available")
        with tab2:
            path=WATERFALL_PATH_TEMPLATE.format(user=user_number)
            if os.path.exists(path):
                st.image(path,use_column_width=True)
            else:
                st.info("Waterfall plot not available")

    # Decision controls
    decision = st.radio("Insider?", ["No","Yes"], index=0, key=f"p{phase_num}_decision")
    confidence = st.radio("Confidence", ["I'm Sure","I'm Not Sure","I Don't Know"], index=0, key=f"p{phase_num}_confidence")
    if decision=="No":
        scenario=0
        st.selectbox("Scenario", [0], format_func=scenario_label, index=0, disabled=True, key=f"p{phase_num}_scenario_no")
    else:
        scenario=st.selectbox("Scenario?", SCENARIO_OPTIONS[1:], format_func=scenario_label, index=0, key=f"p{phase_num}_scenario_yes")

    if (not st.session_state.get(f'phase{phase_num}_done', False)) and st.button(f"Submit Phase {phase_num}", key=f"submit_p{phase_num}"):
        phase_time = time.time() - st.session_state.get(f'p{phase_num}_start', time.time())
        total_time = time.time() - st.session_state.get('total_start', time.time())
        st.session_state[f'phase{phase_num}_time'] = phase_time
        st.session_state[f'phase{phase_num}_total_time'] = total_time

        row = ground_truth_df[ground_truth_df['User Number']==user_number] if ground_truth_df is not None else pd.DataFrame()
        is_mal = (not row.empty) and (row['True Positives Count'].iloc[0]>0) if not row.empty else False
        marks = 1 if ((decision=="Yes")==is_mal) else 0
        case_type = "Without Explainability" if phase_num==1 else "With Explainability"
        save_response(student_id,user_number,scenario,decision,confidence,case_type,marks,phase_time,total_time)

        # Store last TP value for trust score computation
        if not row.empty:
            last_tp = row['True Positives Count'].iloc[0]
        else:
            last_tp = 0
        st.session_state[f'last_tp_user{user_number}_phase{phase_num}'] = last_tp

        
        st.session_state[f'phase{phase_num}_done'] = True
        if f'p{phase_num}_start' in st.session_state:
            del st.session_state[f'p{phase_num}_start']
        if 'total_start' in st.session_state:
            del st.session_state['total_start']

        st.success(f"Phase {phase_num} saved and timer stopped. Questionnaire (below) is not timed.")
    
    # -------- Questionnaire: only show AFTER phase submission (Option B) --------
    if st.session_state.get(f'phase{phase_num}_done', False):
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
            st.success(f"Questionnaire Phase {phase_num} saved")
            st.session_state[f'questionnaire_p{phase_num}_done'] = True



# ==================== MAIN ==================== 
def main(): 
    st.set_page_config( 
        page_title=APP_TITLE, 
        page_icon="🛡️", 
        initial_sidebar_state="collapsed" # Sidebar initially collapsed 
    ) 
    st.title(APP_TITLE) 
    
    # Main radio controls: Instructions | Phase 1 | Phase 2 
    page_choice = st.radio( 
        "Navigation:", 
        ["Instructions", "Phase 1", "Phase 2"], 
        index=0, 
        horizontal=True, 
        key="main_nav" 
    )
    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.header("Setup")
        if st.button("Initialize Databases"):
            # this will DROP and recreate tables (same behavior as your original init_dbs)
            init_dbs()
            st.success("Databases initialized successfully.")
    
    
    st.sidebar.header("Trust Score Calculations & Export")
    st.sidebar.markdown("Compute trust scores for a student/user/phase and export data.")

    # ----- Trust Score Calculation -----
    tin_student = st.sidebar.number_input("Student ID (for trust calc)", min_value=1, max_value=9999, value=1, key="ts_sid")
    tin_user = st.sidebar.number_input("User Number (for trust calc)", min_value=0, max_value=9999, value=0, key="ts_user")
    tin_phase = st.sidebar.selectbox("Phase (for trust calc)", [1,2], index=0, key="ts_phase")

    #if st.sidebar.button("Compute Trust Scores"):
    #log_df = load_log_file()  # read log fresh
    #ts = compute_trust_scores(int(tin_student), int(tin_user), int(tin_phase), log_df)
    #st.sidebar.metric("Subjective Trust", f"{ts['subjective']:.1f}" if ts['subjective'] is not None else "N/A")
    #st.sidebar.metric("Objective Trust", f"{ts['objective']:.1f}" if ts['objective'] is not None else "N/A")
    #st.sidebar.metric("Combined Trust", f"{ts['combined']:.1f}" if ts['combined'] is not None else "N/A")

    
    # ---------------- Trust Score Comparison ----------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Trust Score Comparison")

    # Read log once
    log_df = load_log_file()

    # Compute Phase 1 trust
    ts_phase1 = compute_trust_scores(tin_student, tin_user, 1, log_df)
    combined_phase1 = ts_phase1.get("combined", None)

    # Compute Phase 2 trust
    ts_phase2 = compute_trust_scores(tin_student, tin_user, 2, log_df)
    combined_phase2 = ts_phase2.get("combined", None)

    # Compute change
    if combined_phase1 is not None and combined_phase2 is not None:
        change = combined_phase2 - combined_phase1
    else:
        change = None

    # Display in sidebar
    st.sidebar.metric("Combined Trust Phase 1", f"{combined_phase1:.1f}" if combined_phase1 is not None else "N/A")
    st.sidebar.metric("Combined Trust Phase 2", f"{combined_phase2:.1f}" if combined_phase2 is not None else "N/A")
    st.sidebar.metric("Change in Trust", f"{change:.1f}" if change is not None else "N/A")






    # ----- Data Export -----
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Export")

    # Responses export
    try:
        conn = sqlite3.connect(DB_RESPONSES)
        df_resp = pd.read_sql_query("SELECT * FROM responses ORDER BY timestamp DESC", conn)
        conn.close()
    except Exception:
        df_resp = pd.DataFrame()

    if not df_resp.empty:
        st.sidebar.download_button(
            "Download Phase Results (CSV)", 
            df_resp.to_csv(index=False).encode("utf-8"), 
            file_name="Phase_Results.csv"
        )
    else:
        st.sidebar.write("No phase responses yet")

    # Questionnaire export
    try:
        conn = sqlite3.connect(DB_FEEDBACK)
        df_q = pd.read_sql_query("SELECT * FROM questionnaire_phase ORDER BY timestamp DESC", conn)
        conn.close()
    except Exception:
        df_q = pd.DataFrame()

    if not df_q.empty:
        st.sidebar.download_button(
            "Download Questionnaire (All Phases) (CSV)", 
            df_q.to_csv(index=False).encode("utf-8"), 
            file_name="Questionnaire_All_Phases.csv"
        )
    else:
        st.sidebar.write("No questionnaire entries yet")

    st.sidebar.caption(f"Toronto: {now_toronto().strftime('%Y-%m-%d %I:%M:%S %p')}")


    # Safe initialization at startup: create tables only if missing (DO NOT DROP existing data)
    create_tables_if_missing()

    ground_truth_df = load_ground_truth()
    log_df = load_log_file()


    # Render selected main page
    if page_choice == "Instructions":
        st.header("Instructions")
        st.markdown("""
        ### Overview
        This study has **2 phases**. Each phase simulates analyst review of one user session:
        - **Phase 1:** Without Explainability (AI decision only).
        - **Phase 2:** With Explainability (AI decision + SHAP visuals).

        ### Procedure
        1. Select **Phase 1** or **Phase 2** from the navigation above.
        2. Enter **Student ID** and **User Number**.
        3. The phase timer starts automatically when you open the phase (it records time-on-task).
        4. Review raw data, loss and ROC tabs, then make the decision **Insider? (Yes/No)** and select scenario if Yes.
        5. Click **Submit Phase** to stop the timer — after submission the **Questionnaire** will appear below (no timer).
        6. Complete the questionnaire and submit. Trust scores are computed when both phase and questionnaire are saved.

        ### Timing rules
        - Timers run only during the phase section (not during questionnaires).
        - Phase time is recorded as `phase_time` in the responses DB.
        - `total_time` currently records the same value for single-phase sessions (can be adapted later).

        ### Data & Privacy
        - Responses and questionnaires are stored in local SQLite databases (`student_responses.db`, `feedback.db`).
        - Export CSVs via the sidebar for offline analysis.
        - Do not upload sensitive personal data into these fields.

        ### Analysis notes (for reproducibility)
        - Objective trust uses ground-truth true-positives counts (from `user_true_positives.csv`) and the participant's Yes/No decision.
        - Subjective trust is the mean of questionnaire sliders (scaled to 0–100).
        - Combined trust = 0.6 * subjective + 0.4 * objective when both present.

        If you intend to publish results, include the instrumentation details above in your Methods section.
        """)
        return

    elif page_choice == "Phase 1":
        phase_ui_common(1, log_df, ground_truth_df)
        if st.session_state.get('phase1_done', False):
            student_id = st.session_state.get('student_id', 1)
            user_number = st.session_state.get('phase1_user', 0)
            csv_bytes = questionnaire_phase_to_csv_bytes(student_id, user_number, 1)
            #st.download_button("Download Questionnaire Phase 1 (CSV)", csv_bytes, file_name=f"questionnaire_phase1_user{user_number}.csv")
        return

    elif page_choice == "Phase 2":
        phase_ui_common(2, log_df, ground_truth_df)
        if st.session_state.get('phase2_done', False):
            student_id = st.session_state.get('student_id', 1)
            user_number = st.session_state.get('phase2_user', 0)
            csv_bytes = questionnaire_phase_to_csv_bytes(student_id, user_number, 2)
            st.success("Phase 2 submitted — Questionnaire CSV is available for download below.")
            #st.download_button("Download Questionnaire Phase 2 (CSV)", csv_bytes, file_name=f"questionnaire_phase2_user{user_number}.csv")
        else:
            st.info("After you submit Phase 2, a Download Questionnaire CSV button will appear here.")
        return

if __name__=="__main__":
    main()

