"""Generate Synthetic Data v3 for the Task Delay Risk Predictor.

Same columns as Synthetic_Data_V2. Instead of an AI-made table, each row comes from
simulating the RSF project plan (the RSFS Jira export) 20 times with documented rules.

Usage: python generate_synthetic_v3.py <rsfs_export.csv> <synthetic_v2.csv> <output.csv> [seed] [review|start]

Rules (all parameters are in the PARAMETERS block below):
  * Tasks, story points, roles, milestone gates, predecessors and baseline dates come
    from the RSF plan. They are the same in every run.
  * External_Dependency belongs to the task: 1 if an outside role (sponsor SP,
    privacy/ethics officer PE, store managers SM) is on it. Same in every run.
  * Each run draws the experience of the three team members (PM/PO, DS/SE, DE/MLE).
  * Tasks are processed in dependency order. A late predecessor pushes the task's start,
    minus any slack in the plan; the team recovers half of that push (PASS_ON).
  * Prediction time: each task is rated at the last fortnightly risk review (sprint
    planning) on or before its planned start. Features use only what is visible in Jira
    on that date; the outcome uses what actually happened.
  * Upstream_Delay_Days = the largest known slip among the task's direct predecessors on
    the prediction date: actual slip if done; start delay carried forward if running;
    days past due if still open after its due date; otherwise 0.
  * Assignee_Active_Workload_SP = story points of the other tasks the same person has
    active on the task's expected start date, busiest person on the task, using the
    schedule as known on the prediction date (same definition as the RSFS mapping
    script). Tasks with only outside roles get 0. The delay itself uses the true workload.
  * Plan-structure features at the prediction date (extra columns, added after v3 review):
    Num_Predecessors (direct "blocks" predecessors), Open_Predecessors (not done on the
    prediction date), Slack_Days (days between the latest predecessor's planned finish and
    this task's planned start, capped at 30; 30 if no predecessors). No person-level history
    is used (ethics decision).
  * The task's own delay comes from a written formula plus random noise; blockers, scope
    changes and rare major problems are random events during the task that add days.
  * Actual_Delay_Days = finish slip against the baseline due date (pushed start + own delay).
  * Risk_Outcome: On_Time 0-2 days, Delayed 3-9 days, Missed_Deadline 10+ days (as in v2).
"""
import re
import sys

import numpy as np
import pandas as pd

export_path, v2_path, out_path = sys.argv[1:4]
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 26
PREDICT_AT_ARG = sys.argv[5] if len(sys.argv) > 5 else None

# ---------------- PARAMETERS ----------------
N_RUNS = 20
SNAPSHOT = pd.Timestamp("2027-03-12")            # board status is taken on this date (leaky column, not a feature)
REVIEW_ANCHOR = pd.Timestamp("2027-02-19")       # RSF Sprint 1 start; risk reviews every REVIEW_DAYS from here
PREDICT_AT = "review"                            # "review": sprint planning before the task; "start": the day it starts
SLACK_CAP = 30                                   # Slack_Days is capped; tasks with no predecessors get the cap
REVIEW_DAYS = 14                                 # fortnightly, matching the RSF two-week sprints and status reports
EXTERNAL_ROLES = {"SP", "PE", "SM"}
PERSON = {"PM": "PM/PO", "PO": "PM/PO", "DS": "DS/SE", "SE": "DS/SE", "DE": "DE/MLE", "MLE": "DE/MLE"}
EXPERIENCE_RANGE = (1.0, 9.0)                    # years, drawn per person per run
# Own delay (days) = max(0, linear part + noise + event days)
B0 = -3.5           # base: most tasks finish on time on their own
B_SP = 0.10          # bigger tasks slip more
B_EXT = 1.5          # waiting on sponsor / privacy officer / store managers
B_GATE = 0.8         # approval gates
B_WL = 0.08          # per story point of workload above WL_FREE
WL_FREE = 10         # workload a person can carry without slowing down
B_EXP = -0.25        # per year of experience above the team average (4.5)
NOISE_SD = 2.5
PASS_ON = 0.5        # share of a predecessor's lateness that reaches the next task
                     # (the team recovers the rest by compressing work, as in the RSFS notes)
BLOCKER_RATE = (0.10, 0.35, 0.010)   # Poisson rate = base + ext + per SP
SCOPE_RATE = (0.12, 0.008)           # Poisson rate = base + per SP
DAYS_PER_BLOCKER = 2.0
DAYS_PER_SCOPE_CHANGE = 1.5
MAJOR_ISSUE_RATE = (0.03, 0.05)      # chance of a major problem (rework, failed review) = base + ext
MAJOR_ISSUE_DAYS = (5, 12)           # days it adds, uniform
# --------------------------------------------
if PREDICT_AT_ARG:
    PREDICT_AT = PREDICT_AT_ARG

rng = np.random.default_rng(SEED)
jira = pd.read_csv(export_path)
v2 = pd.read_csv(v2_path)


def labels(row):
    return [row[c] for c in jira.columns if c.startswith("Labels") and pd.notna(row[c])]


def field_date(value):
    return pd.to_datetime(value, format="%d/%b/%y %I:%M %p", errors="coerce")


def baseline_dates(row):
    """Baseline start/finish from the plan line in the description, else the Jira fields."""
    m = re.search(r"Duration: [^(]*\((\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})\)", str(row["Description"]))
    if m:
        return pd.Timestamp(m.group(1)), pd.Timestamp(m.group(2))
    return field_date(row["Custom field (Start date)"]), field_date(row["Due date"])


rows = []
for _, r in jira.iterrows():
    labs = labels(r)
    roles = [x for x in labs if not x.startswith("wbs-") and x not in {"gate", "gate-package", "slipped"}]
    start, finish = baseline_dates(r)
    rows.append({
        "Key": r["Issue key"],
        "Task_ID": r["Issue key"].replace("RSFS-", "RSF-"),
        "Task_Name": re.sub(r"^\d+(\.\d+)*\s+", "", r["Summary"]),
        "Story_Points": float(r["Custom field (Story point estimate)"] or 0),
        "Is_Milestone_Gate": int("gate" in labs),
        "External_Dependency": int(bool(EXTERNAL_ROLES & set(roles))),
        "People": sorted({PERSON[x] for x in roles if x in PERSON}),
        "Preds": [r[c] for c in jira.columns if c.startswith("Inward issue link (Blocks)") and pd.notna(r[c])],
        "B_Start": start,
        "B_Finish": finish,
    })
plan = pd.DataFrame(rows).set_index("Key")
plan["Story_Points"] = plan["Story_Points"].fillna(0)

# Fixed descriptors from v2 (same definitions the project already uses).
fixed = v2.groupby("Task_ID")[["WBS_Code", "Phase", "Task_Category"]].first()
plan = plan.join(fixed, on="Task_ID")
# Two tasks were missing from v2; fill them from the plan.
extra = {"RSF-11": ("1.3.1", "Phase 1: Project Initiation", "Modeling_ML"),
         "RSF-14": ("1.4.2", "Phase 1: Project Initiation", "Project_Management")}
for tid, (wbs, phase, cat) in extra.items():
    m = plan["Task_ID"] == tid
    plan.loc[m, ["WBS_Code", "Phase", "Task_Category"]] = [wbs, phase, cat]

# Dependency order (predecessors first).
order, seen = [], set()
def visit(k):
    if k in seen or k not in plan.index:
        return
    seen.add(k)
    for p in plan.at[k, "Preds"]:
        visit(p)
    order.append(k)
for k in plan.sort_values("B_Start").index:
    visit(k)


def workloads(start, finish):
    """Workload on each task's start day, measured on a given schedule."""
    out = {}
    for k in plan.index:
        best = 0.0
        for person in plan.at[k, "People"]:
            others = [o for o in plan.index if o != k and person in plan.at[o, "People"]
                      and start[o] <= start[k] <= finish[o]]
            best = max(best, sum(plan.at[o, "Story_Points"] for o in others))
        out[k] = best
    return out


def simulate(draws, wl):
    start, finish, upstream, own = {}, {}, {}, {}
    for k in order:
        t = plan.loc[k]
        preds = [p for p in t["Preds"] if p in finish]
        slips = [(finish[p] - plan.at[p, "B_Finish"]).days for p in preds]
        upstream[k] = max([max(s, 0) for s in slips], default=0)
        # Lateness that reaches this task: the predecessor's finish past this task's planned
        # start (slack absorbs the rest), of which the team recovers (1 - PASS_ON).
        reach = [max(0, (finish[p] - t["B_Start"]).days + 1) for p in preds]
        start[k] = t["B_Start"] + pd.Timedelta(days=int(round(PASS_ON * max(reach, default=0))))
        d = draws[k]
        lin = (B0 + B_SP * t["Story_Points"] + B_EXT * t["External_Dependency"] + B_GATE * t["Is_Milestone_Gate"]
               + B_WL * max(0.0, wl[k] - WL_FREE) + B_EXP * (d["exp"] - 4.5) + d["noise"])
        own[k] = max(0, int(round(lin + DAYS_PER_BLOCKER * d["blk"] + DAYS_PER_SCOPE_CHANGE * d["scope"] + d["major"])))
        push = (start[k] - t["B_Start"]).days
        finish[k] = t["B_Finish"] + pd.Timedelta(days=push + own[k])
    return start, finish, upstream


def prediction_date(planned_start):
    """Last fortnightly risk review on or before the task's planned start."""
    if PREDICT_AT == "start":
        return planned_start
    n = (planned_start - REVIEW_ANCHOR).days // REVIEW_DAYS
    return REVIEW_ANCHOR + pd.Timedelta(days=n * REVIEW_DAYS)


def known_slip(p, P, start, finish):
    """How late predecessor p is, as far as anyone can tell on date P."""
    b_start, b_finish = plan.at[p, "B_Start"], plan.at[p, "B_Finish"]
    if finish[p] <= P:                                   # done: its actual slip
        return max(0, (finish[p] - b_finish).days)
    pushed = (start[p] - b_start).days if start[p] <= P else 0   # started late: carried to its finish
    past_due = (P - b_finish).days                       # still open after its due date
    return max(0, pushed, past_due)


def features_at_prediction(start, finish):
    """Upstream delay and workload using only what is visible in Jira on the prediction date."""
    up, wl, extra = {}, {}, {}
    for k in plan.index:
        t = plan.loc[k]
        P = prediction_date(t["B_Start"])
        preds = [p for p in t["Preds"] if p in plan.index]
        slips = {p: known_slip(p, P, start, finish) for p in preds}
        up[k] = max(slips.values(), default=0)
        # Plan-structure features (task-level, not person-level).
        extra[k] = {
            "Num_Predecessors": len(preds),
            "Open_Predecessors": sum(1 for p in preds if finish[p] > P),
            "Slack_Days": (min(SLACK_CAP, max(0, (t["B_Start"] - max(plan.at[p, "B_Finish"] for p in preds)).days - 1))
                           if preds else SLACK_CAP),
        }
        # Expected start as known on P (same recovery rule as the simulation).
        reach = [max(0, (plan.at[p, "B_Finish"] + pd.Timedelta(days=slips[p]) - t["B_Start"]).days + 1) for p in preds]
        day = t["B_Start"] + pd.Timedelta(days=int(round(PASS_ON * max(reach, default=0))))
        best = 0.0
        for person in t["People"]:
            total = 0.0
            for o in plan.index:
                if o == k or person not in plan.at[o, "People"]:
                    continue
                if finish[o] <= P:                       # done: real dates
                    w0, w1 = start[o], finish[o]
                elif start[o] <= P:                      # running: real start, current forecast finish
                    w0 = start[o]
                    w1 = max(plan.at[o, "B_Finish"] + (start[o] - plan.at[o, "B_Start"]), P)
                else:                                    # not started: plan dates
                    w0, w1 = plan.at[o, "B_Start"], plan.at[o, "B_Finish"]
                if w0 <= day <= w1:
                    total += plan.at[o, "Story_Points"]
            best = max(best, total)
        wl[k] = best
    return up, wl, extra


out = []
for run in range(1, N_RUNS + 1):
    exp = {p: round(rng.uniform(*EXPERIENCE_RANGE), 1) for p in ("PM/PO", "DS/SE", "DE/MLE")}
    team_avg = round(np.mean(list(exp.values())), 1)
    draws = {}
    for k in plan.index:
        t = plan.loc[k]
        e = round(np.mean([exp[p] for p in t["People"]]), 1) if t["People"] else team_avg
        draws[k] = {
            "exp": e,
            "noise": rng.normal(0, NOISE_SD),
            "blk": rng.poisson(BLOCKER_RATE[0] + BLOCKER_RATE[1] * t["External_Dependency"] + BLOCKER_RATE[2] * t["Story_Points"]),
            "scope": rng.poisson(SCOPE_RATE[0] + SCOPE_RATE[1] * t["Story_Points"]),
            "major": (rng.uniform(*MAJOR_ISSUE_DAYS)
                      if rng.random() < MAJOR_ISSUE_RATE[0] + MAJOR_ISSUE_RATE[1] * t["External_Dependency"] else 0.0),
        }
    # Pass 1 on the baseline schedule, pass 2 on the run's shifted schedule.
    wl = workloads(plan["B_Start"].to_dict(), plan["B_Finish"].to_dict())
    start, finish, upstream = simulate(draws, wl)
    wl = workloads(start, finish)
    start, finish, upstream = simulate(draws, wl)
    # Features are what the PM would see at the fortnightly review before the task starts;
    # the outcome above used the true workload and true predecessor finishes.
    up_known, wl_known, extra = features_at_prediction(start, finish)

    for k in plan.sort_values("WBS_Code").index:
        t = plan.loc[k]
        delay = (finish[k] - t["B_Finish"]).days
        if finish[k] < SNAPSHOT:
            status = "Done"
        elif start[k] <= SNAPSHOT:
            status = "Blocked" if draws[k]["blk"] > 0 else ("In Review" if rng.random() < 0.3 else "In Progress")
        else:
            status = "To Do"
        out.append({
            "Simulation_Run_ID": f"RUN-{run:02d}",
            "Task_ID": t["Task_ID"],
            "WBS_Code": t["WBS_Code"],
            "Task_Name": t["Task_Name"],
            "Phase": t["Phase"],
            "Task_Category": t["Task_Category"],
            "Current_Board_Status": status,
            "Planned_Due_Date": t["B_Finish"].strftime("%d-%b"),
            "Story_Points": t["Story_Points"],
            "Is_Milestone_Gate": t["Is_Milestone_Gate"],
            "External_Dependency": t["External_Dependency"],
            "Assignee_Active_Workload_SP": round(wl_known[k], 1),
            "Assignee_Experience_Years": draws[k]["exp"],
            "Upstream_Delay_Days": up_known[k],
            "Num_Predecessors": extra[k]["Num_Predecessors"],
            "Open_Predecessors": extra[k]["Open_Predecessors"],
            "Slack_Days": extra[k]["Slack_Days"],
            "Scope_Creep_Changes": draws[k]["scope"],
            "Logged_Blockers": draws[k]["blk"],
            "Actual_Delay_Days": delay,
            "Risk_Outcome": "On_Time" if delay <= 2 else ("Delayed" if delay <= 9 else "Missed_Deadline"),
        })

df = pd.DataFrame(out)
df.to_csv(out_path, index=False)
print(f"{len(df)} rows ({df.Task_ID.nunique()} tasks x {N_RUNS} runs) written to {out_path}")
print(df.Risk_Outcome.value_counts(normalize=True).round(3).to_string())
