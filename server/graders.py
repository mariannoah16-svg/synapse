"""
SYNAPSE v5 — Graders (Final)
==============================
Deterministic graders with improved keyword matching.

Improvements over v4:
  1. Numeric tokens removed from keywords ("max_norm=1.0" → ignore "1.0")
  2. Underscore/equals normalized to spaces ("gradient_clip" = "gradient clip")
  3. Task 3: response_team ONLY — no arbitrary model names
  4. Partial adjacency credit for Task 3 teams (ml vs ai = 0.3)
  5. All scores strictly in [0.0, 1.0]

Test results (all pass):
  "add gradient clipping" vs "add gradient_clip max_norm=1.0" → 0.500 ✅
  "stop training" vs "stop training immediately"               → 1.000 ✅
  "restart printer" vs "reduce batch size"                     → 0.000 ✅
"""
import re
from typing import Dict, List, Set
from models import Action, Reward


def _clamp(v: float) -> float:
    """
    Clamp score to strictly open interval (0.001, 0.999).
    OpenEnv validator requires scores to be strictly between 0 and 1
    — never exactly 0.0 or exactly 1.0.
    """
    return round(max(0.001, min(0.999, v)), 4)


def _keywords(text: str) -> Set[str]:
    """
    Extract meaningful keywords with normalization.
    - Underscores and equals signs → spaces
    - Pure numeric tokens removed (they're noise: 1.0, 32, etc.)
    - Stop words removed
    """
    STOP = {
        "to","the","a","an","and","or","for","of","in","from","with","at","by",
        "is","it","that","this","are","was","be","on","as","up","if","do","not",
        "before","after","then","than","all","any","some","very","just","now",
        "can","will","should","must","make","sure","check","verify","ensure",
        "immediately","properly","correctly","please","your","our","my",
    }
    # Normalize separators
    text = text.lower().replace("_", " ").replace("=", " ").replace(".", " ")
    # Tokenize
    words = set(re.sub(r"[^a-z0-9\s]", " ", text).split())
    # Remove pure numeric tokens
    words = {w for w in words if not re.fullmatch(r"[0-9]+", w)}
    return words - STOP


def _match_score(agent_step: str, correct_step: str) -> float:
    """
    Jaccard keyword similarity between two step descriptions.
    Returns 0.0 – 1.0.
    """
    if agent_step.strip().lower() == correct_step.strip().lower():
        return 1.0
    ak = _keywords(agent_step)
    ck = _keywords(correct_step)
    if not ak or not ck:
        return 0.0
    j = len(ak & ck) / len(ak | ck)
    if j >= 0.7: return round(0.90 + j * 0.10, 4)
    if j >= 0.5: return round(0.60 + j * 0.50, 4)
    if j >= 0.3: return round(j * 1.50, 4)
    if j > 0.0:  return round(j * 0.80, 4)
    return 0.0


def _grade_steps(
    agent_steps: List[str],
    correct_steps: List[str],
    weight: float,
) -> tuple:
    """Grade a list of steps using flexible matching. Returns (score, feedback)."""
    if not agent_steps:
        return 0.0, "no steps provided"
    per_step = weight / len(correct_steps)
    total = 0.0
    hits  = 0
    for cs in correct_steps:
        best = max((_match_score(a, cs) for a in agent_steps), default=0.0)
        total += per_step * best
        if best >= 0.7:
            hits += 1
    return round(total, 4), f"{hits}/{len(correct_steps)} steps matched"


# ── Task 1 — Signal Monitor (Easy) ───────────────────────────────────────────
# Scoring: anomaly_detected(0.20) + anomaly_type(0.50) + severity(0.30)
def grade_task1(action: Action, gt: Dict) -> Reward:
    bd: Dict[str, float] = {}
    fb: List[str] = []

    # 1. Anomaly detected (0.20)
    bd["anomaly_detected"] = 0.20 if action.anomaly_detected == gt["anomaly_detected"] else 0.0
    fb.append(f"{'✅' if bd['anomaly_detected'] else '❌'} anomaly_detected")

    # 2. Anomaly type (0.50)
    bd["anomaly_type"] = 0.50 if action.anomaly_type == gt["anomaly_type"] else 0.0
    fb.append(f"{'✅' if bd['anomaly_type'] else '❌'} type: {action.anomaly_type} vs {gt['anomaly_type']}")

    # 3. Severity (0.30) — partial credit for adjacent level
    SEV = ["low", "medium", "high", "critical"]
    gs, es = action.severity, gt["severity"]
    if gs == es:
        bd["severity"] = 0.30
        fb.append(f"✅ severity: {gs}")
    elif gs in SEV and es in SEV and abs(SEV.index(gs) - SEV.index(es)) == 1:
        bd["severity"] = 0.15
        fb.append(f"⚠️ adjacent severity: {gs} vs {es} (+0.15)")
    else:
        bd["severity"] = 0.0
        fb.append(f"❌ severity: {gs} vs {es}")

    return Reward(score=_clamp(sum(bd.values())), breakdown=bd, feedback=" | ".join(fb))


# ── Task 2 — Root Cause Engine (Medium) ──────────────────────────────────────
# Scoring: root_cause(0.60) + affected_components(0.40, partial per component)
def grade_task2(action: Action, gt: Dict) -> Reward:
    bd: Dict[str, float] = {}
    fb: List[str] = []

    # 1. Root cause (0.60)
    bd["root_cause"] = 0.60 if action.root_cause == gt["root_cause"] else 0.0
    fb.append(
        f"{'✅' if bd['root_cause'] else '❌'} "
        f"root_cause: {action.root_cause} vs {gt['root_cause']}"
    )

    # 2. Affected components (0.40) — partial per correct component
    correct_set = set(gt["affected_components"])
    agent_set   = set(action.affected_components or [])
    if not agent_set:
        bd["components"] = 0.0
        fb.append("❌ no affected_components provided")
    else:
        hits   = len(correct_set & agent_set)
        bd["components"] = round((hits / len(correct_set)) * 0.40, 4)
        wrong  = len(agent_set - correct_set)
        if wrong > 0:
            penalty = round(min(wrong * 0.05, 0.15), 4)
            bd["component_penalty"] = -penalty
            fb.append(f"⚠️ {wrong} extra wrong components (penalty -{penalty})")
        fb.append(f"{'✅' if hits == len(correct_set) else '⚠️'} components: {hits}/{len(correct_set)}")

    return Reward(score=_clamp(sum(bd.values())), breakdown=bd, feedback=" | ".join(fb))


# ── Task 3 — Priority Classifier (Medium-Hard) ───────────────────────────────
# KEY FIX: response_team ONLY. No arbitrary model names.
# Scoring: correct team=1.0, adjacent team=0.3, wrong team=0.0
# This is clean, fully deterministic, and agent CAN know the answer.
def grade_task3(action: Action, gt: Dict) -> Reward:
    TEAMS        = ["ml_team", "ai_team", "devops_team", "all_hands"]
    correct_team = gt["response_team"]
    agent_team   = action.response_team

    if agent_team == correct_team:
        return Reward(
            score=_clamp(1.0),          # 0.999 — strictly < 1.0
            breakdown={"response_team": 0.999},
            feedback=f"✅ Correct team: {agent_team}",
        )
    elif agent_team in TEAMS and correct_team in TEAMS:
        dist    = abs(TEAMS.index(agent_team) - TEAMS.index(correct_team))
        partial = 0.3 if dist == 1 else 0.001   # never exactly 0.0
        return Reward(
            score=_clamp(partial),
            breakdown={"response_team": _clamp(partial)},
            feedback=(
                f"{'⚠️ Adjacent team (+0.3)' if dist == 1 else '❌ Wrong team'}: "
                f"got {agent_team}, expected {correct_team}"
            ),
        )
    else:
        return Reward(
            score=_clamp(0.0),          # 0.001 — strictly > 0.0
            breakdown={"response_team": 0.001},
            feedback=f"❌ Wrong team: got {agent_team}, expected {correct_team}",
        )


# ── Task 4 — Remediation Planner (Hard) ──────────────────────────────────────
# Scoring: immediate(0.35) + root_fix(0.40) + verification(0.25)
# Flexible keyword matching — "stop training" matches "stop training immediately"
def grade_task4(action: Action, gt: Dict) -> Reward:
    bd: Dict[str, float] = {}
    fb: List[str] = []

    phases = [
        ("immediate_steps",    action.immediate_steps,    gt["immediate_steps"],    0.35),
        ("root_fix_steps",     action.root_fix_steps,     gt["root_fix_steps"],     0.40),
        ("verification_steps", action.verification_steps, gt["verification_steps"], 0.25),
    ]
    for pname, asteps, csteps, w in phases:
        if not asteps:
            bd[pname] = 0.0
            fb.append(f"❌ missing {pname}")
        else:
            s, msg = _grade_steps(asteps, csteps, w)
            bd[pname] = s
            fb.append(f"{pname}: {msg}")

    return Reward(score=_clamp(sum(bd.values())), breakdown=bd, feedback=" | ".join(fb))


# ── Task 5 — Post-Mortem Analyst (Very Hard) ─────────────────────────────────
# Scoring: root_cause(0.40) + components(0.20) + prevention(0.20) + monitoring(0.20)
# root_cause = exact match; rest = flexible keyword matching
def grade_task5(action: Action, gt: Dict) -> Reward:
    bd: Dict[str, float] = {}
    fb: List[str] = []

    if not action.postmortem:
        return Reward(
            score=_clamp(0.0),           # 0.001 — strictly > 0.0
            breakdown={"postmortem": 0.001},
            feedback="❌ No postmortem provided",
        )
    pm = action.postmortem

    # 1. Root cause (0.40) — exact
    bd["root_cause"] = 0.40 if pm.get("root_cause") == gt["root_cause"] else 0.0
    fb.append(
        f"{'✅' if bd['root_cause'] else '❌'} "
        f"root_cause: {pm.get('root_cause')} vs {gt['root_cause']}"
    )

    # 2. Components (0.20) — set overlap
    cc = set(gt["affected_components"])
    ac = set(pm.get("affected_components") or [])
    if cc and ac:
        overlap = len(cc & ac) / len(cc)
        bd["components"] = round(0.20 * overlap, 4)
        fb.append(f"{'✅' if overlap==1.0 else '⚠️'} components: {len(cc&ac)}/{len(cc)}")
    else:
        bd["components"] = 0.0
        fb.append("❌ missing components")

    # 3. Prevention steps (0.20) — flexible
    cprev = list(gt["prevention_steps"])
    aprev = list(pm.get("prevention_steps") or [])
    if cprev and aprev:
        s, msg = _grade_steps(aprev, cprev, 0.20)
        bd["prevention"] = s
        fb.append(f"prevention: {msg}")
    else:
        bd["prevention"] = 0.0
        fb.append("❌ missing prevention_steps")

    # 4. Monitoring additions (0.20) — flexible
    cmon = list(gt["monitoring_additions"])
    amon = list(pm.get("monitoring_additions") or [])
    if cmon and amon:
        s, msg = _grade_steps(amon, cmon, 0.20)
        bd["monitoring"] = s
        fb.append(f"monitoring: {msg}")
    else:
        bd["monitoring"] = 0.0
        fb.append("❌ missing monitoring_additions")

    return Reward(score=_clamp(sum(bd.values())), breakdown=bd, feedback=" | ".join(fb))


# ── Registry ──────────────────────────────────────────────────────────────────
GRADERS = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
    "task4": grade_task4,
    "task5": grade_task5,
}

# Calibrated targets — verified against rule-based and LLM agents
BASELINE_TARGETS = {
    "task1": 0.85,   # Easy — clear signals in logs
    "task2": 0.70,   # Medium — needs reasoning
    "task3": 0.80,   # Now easier — response_team only
    "task4": 0.55,   # Hard — multi-step plan
    "task5": 0.45,   # Very hard — structured report
    "average": 0.67, # Ideal research benchmark
}
