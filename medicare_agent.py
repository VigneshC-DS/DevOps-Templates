"""MediCare Patient Follow-Up Agent (LLM decides, code only fetches facts).
Setup: pip install anthropic pandas matplotlib | Keep patient_data.csv in the same folder.
Run: python medicare_agent.py"""

# ======================================================================
# Task 1 - Data Loading & Exploratory Analysis
# ======================================================================
import os, json, getpass
import pandas as pd
import matplotlib.pyplot as plt
from anthropic import Anthropic

if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Anthropic API key: ")  # masked input
client = Anthropic()
MODEL = os.environ.get("MODEL", "claude-sonnet-5")

df = pd.read_csv("patient_data.csv", parse_dates=["last_visit_date", "next_scheduled_visit"])
print("Shape:", df.shape, "| missing values:", df.isna().sum().sum())
# days_since_last_visit was computed against a fixed date, so we recover it (needed to judge "overdue")
AS_OF = (df.last_visit_date + pd.to_timedelta(df.days_since_last_visit, unit="D")).iloc[0]
print("Data snapshot date:", AS_OF.date())
print(df.head())

# ======================================================================
# Task 1 (contd) - EDA
# ======================================================================
conditions = df.diagnosis.str.split(", ").explode().replace({"CKD": "Chronic Kidney Disease"})  # CKD is spelled two ways
print(conditions.value_counts().to_string(), "\n")
print("Missed last appointment:\n", df.missed_last_appointment.value_counts().to_string(), "\n")
print(df.groupby("lab_test").lab_value.agg(["count", "min", "mean", "max"]).round(1))

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
conditions.value_counts().plot.barh(ax=ax[0], title="Patients per condition")
df.boxplot("days_since_last_visit", by="missed_last_appointment", ax=ax[1])
ax[1].set_title("Days since last visit vs missed appointment")
df.boxplot("vitals_spo2", by="missed_last_appointment", ax=ax[2])
ax[2].set_title("SpO2 vs missed appointment")
plt.suptitle(""); plt.tight_layout(); plt.show()
print("Patients whose next visit date is already past the snapshot date:", (df.next_scheduled_visit < AS_OF).sum())

# ======================================================================
# Task 2 - Define Agent Tools
# ======================================================================
# Standard reference ranges given to the LLM as DATA (the LLM interprets the patient's value against them)
LAB_REF = {
 "HbA1c": "Diabetes control: <7% good, 7-9% fair, >9% poor",
 "BNP": "Heart failure: <100 pg/mL normal, 100-400 mild, 400-900 elevated, >900 severe",
 "BP Systolic": "<130 normal, 130-139 stage 1 HTN, 140-179 stage 2 HTN, >=180 hypertensive crisis (mmHg)",
 "FEV1%": "COPD (% predicted): >=80 mild, 50-79 moderate, 30-49 severe, <30 very severe",
 "Hemoglobin": "g/dL: normal >=12 (F) / >=13.5 (M); <10 moderate anaemia; <8 severe",
 "eGFR": "CKD (mL/min/1.73m2): >=90 normal, 60-89 mild, 30-59 moderate, 15-29 severe, <15 kidney failure",
 "TSH": "mIU/L: 0.4-4.0 normal, >4 hypothyroid / under-treated",
 "PHQ-9 Score": "Depression (0-27): <5 minimal, 5-9 mild, 10-14 moderate, 15-19 mod-severe, >=20 severe",
 "Pain Score": "/10: 1-3 mild, 4-6 moderate, 7-10 severe",
 "Peak Flow": "Asthma (L/min): typical adult 400-600; <50% of personal best is danger zone",
 "VITALS": "SpO2 <92% concerning (<90% urgent); resting HR 60-100 bpm",
}
PLANS = {}  # action plans saved by the agent

def search_patients(missed_only=False, min_days_since_visit=0, diagnosis_contains=""):
    d = df[(df.days_since_last_visit >= min_days_since_visit) & df.diagnosis.str.contains(diagnosis_contains, case=False)]
    if missed_only: d = d[d.missed_last_appointment == "Yes"]
    return d[["patient_id", "patient_name", "diagnosis", "days_since_last_visit", "missed_last_appointment"]].to_dict("records")

def get_patient(patient_id):
    r = df[df.patient_id == patient_id]
    if r.empty: return {"error": f"{patient_id} not found"}
    rec = r.iloc[0].to_dict()
    rec["next_visit_overdue_days"] = max(0, (AS_OF - rec["next_scheduled_visit"]).days)
    return {k: (str(v.date()) if isinstance(v, pd.Timestamp) else v) for k, v in rec.items()}

def get_lab_reference(lab_test):
    return {"lab_test": lab_test, "reference": LAB_REF.get(lab_test, "n/a"), "vitals_reference": LAB_REF["VITALS"]}

def save_action_plan(patient_id, risk_level, priority_rank, reasons, actions, outreach_message, needs_clinician_review):
    PLANS[patient_id] = dict(risk_level=risk_level, priority_rank=priority_rank, reasons=reasons, actions=actions,
                             outreach_message=outreach_message, needs_clinician_review=needs_clinician_review)
    return {"status": "saved", "patient_id": patient_id}

FUNCS = dict(search_patients=search_patients, get_patient=get_patient, get_lab_reference=get_lab_reference, save_action_plan=save_action_plan)

S, I, B = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
tool = lambda n, d, p, req=(): {"name": n, "description": d, "input_schema": {"type": "object", "properties": p, "required": list(req)}}
TOOLS = [
 tool("search_patients", "List patients (id, name, diagnosis, days since visit, missed flag). Filter by missed appointments, days since last visit, or diagnosis text.",
      {"missed_only": B, "min_days_since_visit": I, "diagnosis_contains": S}),
 tool("get_patient", "Get the full clinical record of one patient: labs, vitals, medication, notes, visit dates, overdue days.", {"patient_id": S}, ["patient_id"]),
 tool("get_lab_reference", "Get reference ranges for a lab test (and vitals) to interpret a patient's values.", {"lab_test": S}, ["lab_test"]),
 tool("save_action_plan", "Save the final follow-up plan for one patient. priority_rank 1 = most urgent.",
      {"patient_id": S, "risk_level": {"type": "string", "enum": ["Critical", "High", "Medium", "Low"]}, "priority_rank": I,
       "reasons": S, "actions": {"type": "array", "items": S}, "outreach_message": S, "needs_clinician_review": B},
      ["patient_id", "risk_level", "priority_rank", "reasons", "actions", "outreach_message", "needs_clinician_review"]),
]

# ======================================================================
# Task 3 - Agentic Loop
# ======================================================================
SYSTEM = """You are a clinical follow-up assistant for MediCare Clinic's care coordination team.
Use the tools to gather facts; never guess patient data. Judge risk yourself by combining the lab value (vs its reference range),
vitals, diagnosis, medication, free-text notes, missed appointment and overdue follow-up. Weigh clinical severity above mere non-attendance.
You SUPPORT clinicians, you do not diagnose: flag anything serious for clinician review. Be concise."""

def run_agent(task, max_iterations=30, verbose=True):
    messages = [{"role": "user", "content": task}]
    for i in range(1, max_iterations + 1):
        resp = client.messages.create(model=MODEL, max_tokens=4096, system=SYSTEM, tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        calls = [b for b in resp.content if b.type == "tool_use"]
        if not calls:  # no more tool requests -> final answer
            return "".join(b.text for b in resp.content if b.type == "text")
        results = []
        for c in calls:
            try: out = FUNCS[c.name](**c.input)
            except Exception as e: out = {"error": str(e)}  # let the LLM see and recover from errors
            if verbose: print(f"[step {i}] {c.name}({json.dumps(c.input)[:90]}) -> {str(out)[:80]}")
            results.append({"type": "tool_result", "tool_use_id": c.id, "content": json.dumps(out, default=str)})
        messages.append({"role": "user", "content": results})
    return "Stopped: max iterations reached."

# ======================================================================
# Task 4 - Single Patient Analysis
# ======================================================================
PID = "P0008"  # Heart Failure, BNP 1100, dizziness, SpO2 92 - change to any patient_id
answer = run_agent(f"""Analyse patient {PID}. Fetch the record and the lab reference, then reply with ONLY a JSON object:
{{"patient_id":..., "risk_level":"Critical|High|Medium|Low", "reasons":[...], "recommended_actions":[...], "urgency":"e.g. within 24h / 1 week", "needs_clinician_review": true/false}}""")
print("\nFINAL ANSWER:")
try: print(json.dumps(json.loads(answer.strip().strip("`").removeprefix("json").strip()), indent=2))
except Exception: print(answer)

# ======================================================================
# Task 5 - Missed Appointment Follow-Up
# ======================================================================
PLANS.clear()
summary = run_agent("""Find all patients who missed their last appointment. Review EACH one (record + lab reference).
Then rank them from most to least urgent by overall clinical risk and call save_action_plan for every one of them
(priority_rank 1 = most urgent; include a short, friendly outreach message to the patient).
Finish with a 3-line summary for the care manager.""", max_iterations=40)
print("\n" + summary)

# ======================================================================
# Task 5 - Ranked results table
# ======================================================================
plans = pd.DataFrame.from_dict(PLANS, orient="index").rename_axis("patient_id").reset_index().sort_values("priority_rank")
plans = plans.merge(df[["patient_id", "patient_name", "diagnosis"]], on="patient_id")
pd.set_option("display.max_colwidth", 200)
print(plans[["priority_rank", "patient_id", "patient_name", "diagnosis", "risk_level", "reasons", "needs_clinician_review"]].to_string(index=False))

# ======================================================================
# Task 5 - Actions and outreach messages
# ======================================================================
for _, p in plans.iterrows():  # actions and outreach message per patient, in priority order
    print(f"#{p.priority_rank} {p.patient_id} {p.patient_name} [{p.risk_level}]\n  Actions: {'; '.join(p.actions)}\n  Message: {p.outreach_message}\n")
