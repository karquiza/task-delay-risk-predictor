"""GP-25: map the RSFS Jira export to the columns used by the training data.

Usage: python map_rsfs.py <rsfs_export.csv> <synthetic_v2.csv> <output.csv>
"""
import re
import sys
import pandas as pd

export_path, train_path, out_path = sys.argv[1:4]
AS_OF = pd.Timestamp("2027-03-12")          # date of the simulated snapshot
EXTERNAL_ROLES = {"SP", "PE", "SM"}         # Sponsor, Privacy/Ethics officer, Store managers
NOT_ROLES = {"gate", "gate-package", "slipped"}

jira = pd.read_csv(export_path)
train = pd.read_csv(train_path)

def date(col):
    return pd.to_datetime(jira[col], format="%d/%b/%y %I:%M %p", errors="coerce")

def labels(row):
    vals = [row[c] for c in jira.columns if c.startswith("Labels") and pd.notna(row[c])]
    return [v for v in vals if not v.startswith("wbs-")]

def links(row, prefix):
    return [row[c] for c in jira.columns if c.startswith(prefix) and pd.notna(row[c])]

def slip_days(description):
    """Days a task's finish moved in the simulation (revised end - baseline end)."""
    m = re.search(r"baseline \d{4}-\d{2}-\d{2} to (\d{4}-\d{2}-\d{2}); revised \d{4}-\d{2}-\d{2} to (\d{4}-\d{2}-\d{2})",
                  str(description))
    return (pd.Timestamp(m.group(2)) - pd.Timestamp(m.group(1))).days if m else 0

t = pd.DataFrame({
    "Issue_Key": jira["Issue key"],
    "Task_ID": jira["Issue key"].str.replace("RSFS-", "RSF-", regex=False),
    "Task_Name": jira["Summary"],
    "Status": jira["Status"],
    "Start": date("Custom field (Start date)"),
    "Due": date("Due date"),
    "Story_Points": jira["Custom field (Story point estimate)"].fillna(0),
})
t["Roles"] = [[r for r in labels(row) if r not in NOT_ROLES] for _, row in jira.iterrows()]
t["Is_Milestone_Gate"] = [int("gate" in labels(row)) for _, row in jira.iterrows()]
t["Predecessors"] = [links(row, "Inward issue link (Blocks)") for _, row in jira.iterrows()]
t["Slip_Days"] = jira["Description"].map(slip_days)

# 1. Fixed task attributes come from the training data (same definitions as the model learned).
fixed = train.groupby("Task_ID")[["WBS_Code", "Phase", "Task_Category"]].first()
t = t.join(fixed, on="Task_ID")

# 2. External dependency: an outside role (sponsor, privacy officer, store managers) is involved.
t["External_Dependency"] = t["Roles"].map(lambda rs: int(bool(EXTERNAL_ROLES & set(rs))))

# 3. Workload: Jira has roles, not people, so roles are mapped to team members as in Lab 6.
#    Workload = story points of the OTHER tasks the same person has active on the day this
#    task starts (the training data uses the owner's workload "when the task begins"), including
#    tasks that are Done by now but were running on that day. A task with several people takes
#    the busiest one.
PERSON = {"PM": "PM/PO", "PO": "PM/PO", "DS": "DS/SE", "SE": "DS/SE", "DE": "DE/MLE", "MLE": "DE/MLE"}
t["People"] = t["Roles"].map(lambda rs: sorted({PERSON[r] for r in rs if r in PERSON}))
def workload(row):
    best = 0.0
    for person in row["People"]:
        active = t[(t["Issue_Key"] != row["Issue_Key"])
                   & t["People"].map(lambda ps: person in ps)
                   & (t["Start"] <= row["Start"]) & (t["Due"] >= row["Start"])]
        best = max(best, active["Story_Points"].sum())
    return best
t["Assignee_Active_Workload_SP"] = t.apply(workload, axis=1)

#    Adjustments so values stay within what the model learned (training range 8-42):
#    - no team member on the task (outside roles only): workload unknown -> training median
#      (external involvement is already captured by External_Dependency)
#    - team task below the training minimum: set to the minimum ("lightly loaded")
WL_MIN = train["Assignee_Active_Workload_SP"].min()
WL_MEDIAN = round(train["Assignee_Active_Workload_SP"].median(), 1)
no_team = t["People"].map(len) == 0
low = ~no_team & (t["Assignee_Active_Workload_SP"] < WL_MIN)
t["Workload_Source"] = "calculated"
t.loc[no_team, "Workload_Source"] = "no team owner: training median"
t.loc[low, "Workload_Source"] = "below training range: set to minimum"
t.loc[no_team, "Assignee_Active_Workload_SP"] = WL_MEDIAN
t.loc[low, "Assignee_Active_Workload_SP"] = WL_MIN

# 4. Upstream delay: largest slip among the task's direct predecessors.
slip = dict(zip(t["Issue_Key"], t["Slip_Days"]))
t["Upstream_Delay_Days"] = t["Predecessors"].map(lambda ps: max([slip.get(p, 0) for p in ps], default=0))

# 5. Experience: not recorded in Jira -> training-data median for every task (stated limitation).
t["Assignee_Experience_Years"] = round(train["Assignee_Experience_Years"].median(), 1)

cols = ["Issue_Key", "Task_ID", "WBS_Code", "Task_Name", "Phase", "Task_Category", "Status",
        "Story_Points", "Is_Milestone_Gate", "External_Dependency", "Assignee_Active_Workload_SP",
        "Assignee_Experience_Years", "Upstream_Delay_Days", "Workload_Source"]
out = t[t["Status"] != "Done"][cols]
out.to_csv(out_path, index=False)
print(f"{len(out)} open tasks written to {out_path}")
print(out[["Story_Points", "External_Dependency", "Assignee_Active_Workload_SP", "Upstream_Delay_Days"]].describe().round(1).to_string())
missing = out[out["Task_Category"].isna()]["Issue_Key"].tolist()
if missing:
    print("Not in training data (no category/phase):", missing)
