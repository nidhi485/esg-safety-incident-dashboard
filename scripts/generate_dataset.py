"""Generate a SYNTHETIC extended incident dataset (seeded, reproducible).

Why: the bundled CSV has only 6 rows - too few to train or validate anything.
This script expands it with template-based narratives. Every row carries a
`template_id` so that validation can hold out whole templates (GroupKFold)
instead of leaking near-duplicate sentences between train and test.

IMPORTANT: this is synthetic data written by the project author. Metrics on it
measure "can the model learn these label conventions", NOT real-world accuracy.

Usage:  python scripts/generate_dataset.py
Output: data/extended_incidents.csv
"""
import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
OUT = Path(__file__).resolve().parents[1] / "data" / "extended_incidents.csv"

SLOTS = {
    "zone": ["solar block 3", "solar block 7", "inverter yard", "wind turbine pad T12", "wind turbine pad T27",
             "substation yard", "cable trench corridor", "contractor storage area", "main gate", "O&M workshop",
             "penstock access road", "powerhouse deck", "BESS container area"],
    "equip": ["wind gearbox", "power transformer", "inverter skid", "DG set", "hydraulic unit", "tracker motor",
              "switchgear panel", "battery rack"],
    "worker": ["contractor technician", "O&M operator", "electrician", "panel cleaning crew member",
               "security guard", "rigging crew member"],
    "task": ["panel cleaning", "cable laying", "blade inspection", "transformer maintenance", "module installation",
             "yard grading", "gearbox oil top-up", "vegetation clearing"],
    "shift": ["night shift", "early morning shift", "second shift", "weekend shift"],
    "vehicle": ["pickup truck", "water tanker", "crane", "forklift", "tractor trolley"],
    "surface": ["wet walkway", "oily floor", "muddy access road", "freshly washed ramp"],
    "ppe": ["safety helmet", "gloves", "safety shoes", "eye protection", "high-visibility vest"],
    "permit": ["consent to operate", "fire NOC", "hazardous waste authorisation", "work-at-height permit register",
               "pollution control board consent"],
    "topic": ["waste handling", "contractor safety induction", "water stewardship", "biodiversity", "emergency drill"],
    "issue": ["dust from haul roads", "noise at night", "grazing land access", "local hiring", "road damage"],
    "register": ["visitor", "PPE issue", "calibration", "waste manifest"],
    "leak": ["valve leakage", "hose pipe left running", "tank overflow", "broken coupling"],
}

# (template_id, category, severity, [phrasing variants], rising_trend)
# rising_trend: +1 more frequent later in the year, -1 less frequent, 0 flat
T = [
    # ---------------- SAFETY ----------------
    ("SF01", "safety", "high", [
        "Near miss: {vehicle} movement close to {worker} near {zone} during {shift}",
        "Near miss as {vehicle} reversed without a banksman towards workers at {zone}, {shift}"], 0),
    ("SF02", "safety", "high", [
        "{worker} received an electric shock while isolating {equip}; lockout tagout step skipped",
        "Electric shock to {worker} at {zone}, isolation of {equip} not verified before work"], 0),
    ("SF03", "safety", "high", [
        "Fall from height by {worker} during {task} at {zone}, no fall arrest anchored",
        "{worker} working at height without harness during {task}, found by supervisor at {zone}"], 0),
    ("SF04", "safety", "high", [
        "Small fire in {equip} cabinet detected by smoke alarm at {zone}, extinguished by operator",
        "Sparking and smoke from {equip} at {zone}, fire extinguisher used to put out flames"], 0),
    ("SF05", "safety", "medium", [
        "{worker} slipped on {surface} near {zone}, minor injury, first aid given",
        "Contractor slipped near {surface} at {zone}, minor injury, housekeeping issue"], 0),
    ("SF06", "safety", "medium", [
        "PPE non-compliance observed: {worker} without {ppe} during {task}",
        "PPE non-compliance observed during {task}, {ppe} not worn at {zone}"], 1),
    ("SF07", "safety", "medium", [
        "Hand cut while {task} at {zone}, first aid treatment, unsuitable tool used",
        "{worker} sustained a minor hand laceration during {task}, first aid given"], 0),
    ("SF08", "safety", "medium", [
        "Heat stress symptoms reported by {worker} during {task} in afternoon, rest and water provided",
        "{worker} felt dizzy from heat during {task} at {zone}, moved to shade and given first aid"], 1),
    ("SF09", "safety", "low", [
        "Housekeeping deviation: loose cables and packaging left in {zone}",
        "Untidy work area at {zone}, tools and cable offcuts lying on the walkway, cleaned on the spot"], 1),
    ("SF10", "safety", "low", [
        "Unsafe act observation: {worker} not using handrail on stairs at {zone}, corrected on the spot",
        "Observation: {worker} walking without holding handrail at {zone}, counselled immediately"], 0),
    ("SF11", "safety", "low", [
        "Toolbox talk record missing for {shift} at {zone}",
        "Daily toolbox talk not documented for {task} crew at {zone}"], 0),
    ("SF12", "safety", "medium", [
        "Dropped object from height landed inside exclusion zone at {zone}, no one injured, area barricaded",
        "Tool fell during {task} near {zone}, landed close to ground crew, work stopped for review"], 0),
    # ---------------- ENVIRONMENT ----------------
    ("EN01", "environment", "high", [
        "Oil leak from {equip} reached the drain outside containment bund at {zone}",
        "Transformer oil spill reaching soil near {zone}, containment breached, spill response activated"], 0),
    ("EN02", "environment", "high", [
        "Diesel spill during {equip} refuelling at {zone} entered storm water drain",
        "Fuel spill from {equip} at {zone} flowed towards nearby stream, booms deployed"], 0),
    ("EN03", "environment", "medium", [
        "Oil seepage observed near {equip} at {zone}, contained with spill kit",
        "Minor oil seepage from {equip} found during inspection of {zone}, absorbent pads applied"], 0),
    ("EN04", "environment", "medium", [
        "Water usage spike during {task} due to {leak} at {zone}",
        "Water usage spike during module cleaning due to {leak}, tank level dropped overnight"], 1),
    ("EN05", "environment", "medium", [
        "Sewage treatment plant at {zone} exceeded effluent discharge limit in routine sample test",
        "Effluent sample from {zone} above permissible limit, retest ordered"], 0),
    ("EN06", "environment", "medium", [
        "Dust from unpaved road at {zone} affecting nearby village, complaint received",
        "Excess dust during {task} near {zone}, water sprinkling delayed, villagers complained"], 0),
    ("EN07", "environment", "medium", [
        "Multiple bird carcasses found near {zone} during morning inspection",
        "Bat and bird collision remains found around turbine area {zone} after night operation"], 0),
    ("EN08", "environment", "low", [
        "Minor water leakage from tap at {zone} wash area, repaired same day",
        "Dripping water line at {zone}, plumber fixed it within hours"], 0),
    ("EN09", "environment", "low", [
        "Used oil drum stored without drip tray at {zone}",
        "Oil drums kept on bare ground near {zone}, drip tray not provided"], -1),
    ("EN10", "environment", "low", [
        "Vehicle idling for long periods at {zone}, observed during gate audit",
        "Excess engine idling at {zone} reported by security"], 0),
    ("EN11", "environment", "low", [
        "Noise slightly above internal target near {equip} during test run at {zone}",
        "Temporary noise increase near {equip} at {zone}, within regulatory limit"], 0),
    ("EN12", "environment", "high", [
        "Battery electrolyte leak from {equip} at {zone} reached ground outside bunded area",
        "Coolant release from {equip} at {zone} escaped secondary containment"], 0),
    # ---------------- ESG COMPLIANCE ----------------
    ("EC01", "esg_compliance", "high", [
        "Statutory inspection found expired {permit} at {zone}, regulator notice issued",
        "Regulator issued notice after {permit} lapsed without renewal at {zone}"], 0),
    ("EC02", "esg_compliance", "high", [
        "Labour inspector flagged wage record discrepancy for contractor workers at {zone}",
        "Contractor minimum wage records inconsistent for site workers, inspector raised non-conformity"], 0),
    ("EC03", "esg_compliance", "high", [
        "Community grievance on {issue} escalated to local authority near {zone}",
        "Villagers escalated {issue} complaint to district office, site operations questioned"], 0),
    ("EC04", "esg_compliance", "high", [
        "Hazardous waste handed to unauthorised vendor, manifest missing at {zone}",
        "Used oil removed from {zone} by vendor without valid authorisation or manifest"], 0),
    ("EC05", "esg_compliance", "medium", [
        "Audit finding on {topic} still open beyond due date at {zone}",
        "Corrective action for {topic} audit observation overdue by several weeks"], 0),
    ("EC06", "esg_compliance", "medium", [
        "Contractor {topic} training records incomplete for several workers at {zone}",
        "Safety induction records missing for new contractor crew at {zone}"], 1),
    ("EC07", "esg_compliance", "medium", [
        "Community grievance on {issue} logged, response pending beyond 7 days",
        "Grievance about {issue} from nearby village not yet closed, follow-up overdue"], 1),
    ("EC08", "esg_compliance", "medium", [
        "Environmental return for {permit} submitted late for the quarter",
        "Statutory compliance report for {permit} filed after the due date"], 0),
    ("EC09", "esg_compliance", "low", [
        "Repeated waste segregation deviation at {zone}",
        "Waste segregation not followed again at {zone}, mixed bins noticed by auditor"], 1),
    ("EC10", "esg_compliance", "low", [
        "Signage for {topic} policy missing at {zone} entrance",
        "Policy display board on {topic} not updated at {zone}"], 0),
    ("EC11", "esg_compliance", "low", [
        "Minor gap in {register} register at {zone}, corrected during review",
        "{register} register entries incomplete for one day, updated after reminder"], 0),
    ("EC12", "esg_compliance", "low", [
        "CSR visitor log not updated for the month at {zone}",
        "Stakeholder visit record missing for one site visit at {zone}"], 0),
]


def fill(tpl, rng):
    out = tpl
    for key, vals in SLOTS.items():
        token = "{" + key + "}"
        while token in out:
            out = out.replace(token, rng.choice(vals), 1)
    return out


def main():
    rng = random.Random(SEED)
    months = list(range(1, 10))  # Jan..Sep 2026
    rows, n = [], 1
    per_month = 18
    for m in months:
        weights = []
        for (_, _, _, _, trend) in T:
            weights.append(max(0.15, 1.0 + 0.22 * trend * (m - 5)))
        for _ in range(per_month):
            tid, cat, sev, variants, _ = rng.choices(T, weights=weights, k=1)[0]
            text = fill(rng.choice(variants), rng)
            text = text[0].upper() + text[1:]
            day = rng.randint(1, 28)
            rows.append({
                "incident_id": f"X{n:03d}", "narrative": text, "target_category": cat, "target_severity": sev,
                "template_id": tid, "incident_date": date(2026, m, day).isoformat(),
                "site_type": rng.choice(["solar", "wind", "hydro", "bess_substation"]),
                "source": "synthetic",
            })
            n += 1
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
