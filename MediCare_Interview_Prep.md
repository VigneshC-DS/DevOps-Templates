# MediCare Patient Follow-Up Agent: Interview Prep Guide

## 1. The problem in simple words

MediCare Clinic looks after 1,000+ chronic disease patients (diabetes, hypertension, heart failure, COPD, anemia and more). Problems are usually found only at the next scheduled visit. By then, a patient who missed an appointment or has worsening labs may already have a complication or a hospital readmission.

**The ask:** build a prototype AI agent that reads patient records on its own, finds who is at risk, and tells the care team who to contact first and what to do.

**Key rule from the brief:** the LLM must make the decisions. Risk logic must not be hardcoded.

**Constraints:** one Jupyter notebook, runs top to bottom, all outputs visible, under 250 lines of code, real LLM API, 40 minutes.

---

## 2. The dataset in one page

| Item | Detail |
|---|---|
| Size | 100 patients, 18 columns, one row per patient (latest snapshot) |
| Quality | No missing values, unique patient IDs |
| Snapshot date | 2024-12-31 (recovered from `last_visit_date + days_since_last_visit`) |
| Missed last appointment | 15 Yes, 85 No |
| Overdue follow-up | 97 of 100 have a next scheduled visit that is already in the past |

**Column groups**
- **Identity:** patient_id, name, age, gender
- **Clinical:** diagnosis (can be compound, e.g. "Type 2 Diabetes, CKD"), current_medication
- **Lab:** lab_test, lab_value, lab_unit (one lab per patient)
- **Vitals:** BP systolic and diastolic, heart rate, SpO2 (SpO2 range is 90 to 99)
- **Follow-up:** last_visit_date, next_scheduled_visit, days_since_last_visit, missed_last_appointment
- **Free text:** notes (only 10 distinct phrases, e.g. "dizziness and fatigue", "missed doses")

**Patients per condition (compound diagnoses split)**

| Condition | Patients |
|---|---|
| Hypertension | 25 |
| Heart Failure | 23 |
| Type 2 Diabetes | 17 |
| Anemia | 14 |
| COPD | 13 |
| Chronic Kidney Disease (incl. "CKD") | 9 |
| Hypothyroidism | 9 |
| Osteoarthritis | 7 |
| Depression | 7 |
| Asthma | 6 |

**Labs in the data:** HbA1c, BNP, BP Systolic, FEV1%, Hemoglobin, eGFR, TSH, PHQ-9, Pain Score, Peak Flow.

**Data quality points to mention**
1. Only one lab per patient, so there is no trend information.
2. The lab does not always match the diagnosis, so the agent must read labs together with vitals and notes.
3. "CKD" and "Chronic Kidney Disease" are the same condition. I merged them.
4. Some rows look unrealistic (a male patient named Priya, a 29-year-old with heart failure). The data is synthetic.
5. `days_since_last_visit` was calculated against a fixed date, so I recovered that date to judge overdue visits.

---

## 3. What I built (task by task)

| Task | Implementation |
|---|---|
| 1. Data and EDA | Load CSV, condition counts, missed-appointment rate, lab summary, 3 charts, data quality notes |
| 2. Tools | `search_patients`, `get_patient`, `get_lab_reference`, `save_action_plan` |
| 3. Agent loop | Send task and tools to the LLM, run requested tools, return results, repeat until a final answer. Has a max-iteration cap, error handling and a printed trace |
| 4. Single patient | P0008 analysed by the agent. Returns JSON with risk level, reasons, actions, urgency, clinician-review flag |
| 5. Missed appointments | Agent finds the 15 patients, reviews each, ranks by clinical risk, saves a plan and an outreach message per patient. Results shown in a table |

**Design principle:** tools fetch facts, and the LLM makes judgements. Reference ranges are given to the LLM as data, and it interprets the patient's value against them.

**Safety:** the system prompt says the agent supports clinicians and does not diagnose. Each plan has a `needs_clinician_review` flag.

---

## 4. Interview questions with model answers

### About the data

**Q1. What is in your dataset and what quality issues did you find?**
100 patient snapshots with demographics, diagnosis, medication, one lab result, vitals, visit dates and free-text notes. There are no missing values. Issues: only one lab per patient, CKD spelled two ways, some unrealistic demographics because the data is synthetic, and many visits already overdue.

**Q2. How did you handle compound diagnoses?**
I split them on commas before counting, so a patient with "Diabetes, CKD" counts under both. I also merged "CKD" and "Chronic Kidney Disease". For the agent, the full diagnosis string is passed as-is so the LLM sees the combined picture.

**Q3. Why is one lab per patient a limitation?**
I can't see whether a value is improving or worsening, which matters clinically. A next step would be adding lab history so the agent can reason about trends.

**Q4. What insight did you find in the EDA?**
15% of patients missed their last appointment, 97% have an overdue next visit, and the free-text notes contain clinical signals like dizziness and missed doses that rules would struggle to weigh together with labs and vitals.

### About the design

**Q5. What is an AI agent, and how is it different from a single prompt?**
A single prompt gets one answer from the model. An agent runs in a loop: the model decides which tool to call, the code runs it, the result goes back, and the model decides the next step until the task is done.

**Q6. Why let the LLM decide risk instead of if/else rules?**
The brief requires it. It is also better clinically, because risk depends on many things together (lab, vitals, diagnosis, notes, missed visit). Rules with fixed thresholds miss combinations such as a "stable" lab with worrying symptoms in the notes. The LLM can also explain its reasoning.

**Q7. Why these four tools?**
Together they cover discovery (`search_patients`), detail (`get_patient`), knowledge (`get_lab_reference`) and action (`save_action_plan`). Each returns facts or stores results and contains no risk verdicts.

**Q8. Why give reference ranges as data instead of coding verdicts?**
If I coded "HbA1c > 9 means high risk", that is hardcoded logic. Giving the range as information lets the LLM decide how it combines with everything else.

**Q9. Why only analyse the missed-appointment patients in Task 5?**
That is what the task asks, and it is efficient: 15 patients instead of 100 LLM-driven reviews. For all 100, I would filter first and batch.

### About the agent loop

**Q10. How does the loop know when to stop?**
When the model replies without requesting any tool, that reply is the final answer. There is also a maximum iteration cap as a safety net.

**Q11. What if a tool fails or the model calls the wrong tool?**
Errors are caught and returned to the model as a tool result, so it can correct itself. Unknown patient IDs return an error message instead of crashing.

**Q12. How do you reduce hallucination?**
The system prompt says never guess patient data and to fetch facts via tools. Every claim comes from tool output. In production I would also validate the output format and log traces.

### About healthcare and safety

**Q13. Why rank by clinical risk instead of who missed?**
A stable patient who missed a routine check-up is less urgent than a heart failure patient with a high BNP and low SpO2 who also missed. Missing an appointment is one signal, not the whole picture.

**Q14. Is it safe to let AI make these decisions?**
It is decision support, not diagnosis. Every plan carries a clinician-review flag and a human makes the final call. I would also audit outputs against clinician triage before any real use.

**Q15. What about privacy?**
The data is anonymised. In production I would remove names and identifiers before sending data to an external API, or use a compliant, private deployment.

### Scaling and next steps

**Q16. How would you scale to 1,000+ patients?**
Pre-filter by simple facts (missed, overdue), batch the rest, run the agent in parallel, cache reference data, and schedule a daily run.

**Q17. How would you evaluate it?**
Have clinicians triage a sample and measure agreement with the agent's risk levels. Check that critical cases are never under-ranked, and track consistency by running the same patient several times.

**Q18. What would you improve with more time?**
Add lab history for trends, run on all patients in batches, add structured output validation, log every agent trace for audit, and build a dashboard for the care team.

**Q19. What are the limitations of your prototype?**
One lab per patient, synthetic notes with few distinct phrases, results can vary between runs, and no clinical validation yet.

**Q20. Why did you keep the code so short?**
The brief caps it at 250 lines. My notebook uses about 106 lines by keeping tool definitions compact and reusing a single agent loop for both tasks.

---

## 5. Quick tips for the interview

- Start with the one-line summary: "A digital care coordinator that reads patient data, reasons about risk and tells staff who to call first and why."
- Show one example patient end to end (Task 4), then the ranked table (Task 5).
- Be honest that the data is small and synthetic, and say what you would do next.
- Emphasise that the LLM decides and the code only fetches facts, since that is the core requirement.
- Before the interview, run the notebook once with your API key so all outputs are visible and saved.
