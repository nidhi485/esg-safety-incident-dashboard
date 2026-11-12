"""Rule-based NLP helpers: text normalisation, hazard tagging, severity cues,
structured extraction and single-incident summary.

Everything here is deterministic and inspectable. The lexicons were written by
the project author (not learned), so they encode one person's reading of the
domain and need EHS-team review before any real use.
"""
import re

# ---------------------------------------------------------------- normalisation
def normalize(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\s\-']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------- hazard tags
# tag -> (human label, category it usually belongs to, regex)
HAZARD_TAGS = {
    "fall_from_height": ("Fall from height", "safety", r"fall from height|at height|fall arrest|harness|dropped object|tool fell"),
    "electrical": ("Electrical hazard", "safety", r"\belectric(al|ity)?\b|\bshock\b|lockout|isolat|energi[sz]ed|arc flash|sparking"),
    "fire": ("Fire / smoke", "safety", r"\bfire\b(?! noc)|smoke|flame|extinguish"),
    "vehicle_movement": ("Vehicle movement", "safety", r"\b(vehicle|reversed|truck|forklift|crane|tractor|tanker)\b|banksman"),
    "slip_trip": ("Slip / trip", "safety", r"\bslip(ped)?\b|\btrip(ped)?\b|wet walkway|oily floor|handrail"),
    "ppe": ("PPE compliance", "safety", r"\bppe\b|helmet|gloves|safety shoes|eye protection|high-visibility|harness not"),
    "heat_stress": ("Heat stress", "safety", r"heat stress|dizzy|heat"),
    "housekeeping": ("Housekeeping", "safety", r"housekeeping|loose cable|packaging|untidy|offcuts"),
    "manual_handling_injury": ("Hand / tool injury", "safety", r"hand cut|laceration|unsuitable tool"),
    "oil_fuel_spill": ("Oil / fuel release", "environment", r"\boil\b|diesel|fuel|spill|seepage|coolant|electrolyte"),
    "water_use": ("Water use / leakage", "environment", r"water usage|water leak|valve|tank overflow|hose|dripping|\btap\b"),
    "effluent": ("Effluent / discharge", "environment", r"effluent|sewage|discharge"),
    "dust_noise": ("Dust / noise", "environment", r"dust|noise"),
    "biodiversity": ("Biodiversity impact", "environment", r"\bbirds?\b|\bbats?\b|carcass|wildlife|collision remains"),
    "emissions_idling": ("Vehicle idling", "environment", r"idling"),
    "waste_mgmt": ("Waste management", "esg_compliance", r"waste|segregat|manifest|drum"),
    "statutory_permit": ("Statutory permit / regulator", "esg_compliance", r"regulator|statutory|permit|consent|licen[cs]e|noc\b|authori[sz]ation|inspector|return for|compliance report"),
    "labour_contractor": ("Labour / contractor records", "esg_compliance", r"wage|contractor .*records|training records|induction records|labour"),
    "community": ("Community grievance", "esg_compliance", r"grievance|villag|community|escalated"),
    "audit_documentation": ("Audit / documentation gap", "esg_compliance", r"audit|register|signage|display board|toolbox|record|log not|policy"),
}
_COMPILED = {k: re.compile(v[2]) for k, v in HAZARD_TAGS.items()}

# Priority order when picking ONE primary tag (more specific / more severe first).
TAG_PRIORITY = ["electrical", "fall_from_height", "fire", "vehicle_movement", "oil_fuel_spill", "biodiversity",
                "effluent", "statutory_permit", "community", "labour_contractor", "waste_mgmt", "slip_trip", "ppe",
                "heat_stress", "manual_handling_injury", "housekeeping", "water_use", "dust_noise",
                "emissions_idling", "audit_documentation"]


def hazard_tags(text: str):
    t = normalize(text)
    found = [k for k, rx in _COMPILED.items() if rx.search(t)]
    return sorted(found, key=lambda k: TAG_PRIORITY.index(k))


def primary_tag(text: str):
    tags = hazard_tags(text)
    return tags[0] if tags else "other"


def tag_label(tag: str) -> str:
    return HAZARD_TAGS.get(tag, ("Other / unclassified", "", ""))[0]


# ---------------------------------------------------------------- severity cues
HIGH_CUES = [
    r"electric shock", r"fall from height", r"without harness", r"\bfire\b(?! noc).*(extinguish|alarm)|smoke alarm",
    r"fatal", r"hospital", r"lost[- ]time", r"near miss.*(vehicle|crane|forklift|truck)",
    r"\b(vehicle|truck|crane|forklift|tanker)\b.*(close to|towards) .*(worker|trench)|close to cable trench",
    r"outside (the )?(containment|bund)|beyond (the )?bund|breached|entered .*drain|storm water drain|towards nearby stream",
    r"regulator notice|notice issued|non-conformity|escalated to (local authority|district)|unauthori[sz]ed|without valid authori",
    r"wage record|manifest missing|without .*manifest", r"lapsed|expired",
]
LOW_CUES = [
    r"observation", r"corrected on the spot|counselled|cleaned on the spot", r"repaired same day|fixed it within",
    r"not documented|record missing|log not|not updated|minor gap|entries incomplete|signage|display board",
    r"without (a )?drip tray|drip tray not", r"slightly above|within regulatory", r"idling", r"toolbox",
    r"repeated waste segregation|segregation not followed|mixed bins",
]
_HIGH = [re.compile(p) for p in HIGH_CUES]
_LOW = [re.compile(p) for p in LOW_CUES]


def severity_cues(text: str):
    """Return (high_cues_hit, low_cues_hit) as lists of the matched phrases from the narrative."""
    t = normalize(text)
    def hits(rxs):
        out = []
        for p in rxs:
            m = p.search(t)
            if m and m.group(0).strip() not in out:
                out.append(m.group(0).strip())
        return out
    return hits(_HIGH), hits(_LOW)


def rule_severity(text: str):
    """Pure rule engine. Returns (label, evidence_list)."""
    hi, lo = severity_cues(text)
    if hi:
        return "high", ["high cue: " + h for h in hi]
    if lo:
        return "low", ["low cue: " + l for l in lo]
    return "medium", ["no high or low cue matched -> default medium"]


# ---------------------------------------------------------------- structured extraction
LOCATION_TERMS = [
    "solar block", "inverter yard", "wind turbine pad", "turbine", "substation yard", "substation", "cable trench",
    "contractor storage area", "storage area", "main gate", "gate", "o&m workshop", "workshop", "penstock",
    "powerhouse deck", "powerhouse", "bess container area", "bess", "gearbox maintenance bay", "maintenance bay",
    "walkway", "nacelle", "drain", "stream", "wash area", "entrance",
]
CONSEQUENCE_TERMS = [
    "minor injury", "first aid", "laceration", "electric shock", "fall from height", "fire", "smoke",
    "regulator notice", "notice issued", "complaint", "escalated", "carcass", "exceeded", "above permissible",
    "reached the drain", "reached soil", "entered storm water drain", "outside containment", "water usage spike",
    "no one injured", "near miss", "overdue", "late", "missing", "incomplete", "expired", "lapsed",
]
CONTROL_TERMS = [
    "contained with spill kit", "spill kit", "absorbent pads", "booms deployed", "spill response activated",
    "extinguished", "fire extinguisher", "corrected on the spot", "counselled", "repaired same day", "fixed",
    "first aid given", "first aid treatment", "moved to shade", "barricaded", "work stopped", "retest ordered",
    "rest and water provided", "cleaned on the spot", "updated after reminder", "corrected during review",
]


def _find(text, terms):
    t = normalize(text)
    hits = []
    for term in terms:
        if term in t and not any(term in h for h in hits):
            hits.append(term)
    return hits


def extract_structure(text: str):
    return {
        "locations": _find(text, LOCATION_TERMS)[:3],
        "consequences": _find(text, CONSEQUENCE_TERMS)[:4],
        "controls_already_applied": _find(text, CONTROL_TERMS)[:3],
    }


def summarize_incident(text: str, category: str, severity: str, tag: str):
    """Extractive/template summary: every phrase comes from the narrative or a lexicon."""
    s = extract_structure(text)
    parts = [f"{severity.upper()}-severity {category.replace('_', ' ')} incident ({tag_label(tag).lower()})"]
    if s["locations"]:
        parts.append("at " + ", ".join(s["locations"][:2]))
    line = " ".join(parts)
    if s["consequences"]:
        line += "; indicators: " + ", ".join(s["consequences"])
    if s["controls_already_applied"]:
        line += "; already done: " + ", ".join(s["controls_already_applied"])
    else:
        line += "; no immediate control recorded in the narrative"
    return line + "."
