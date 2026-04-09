"""
SYNAPSE — Graders (Final, Validator-Compliant)
================================================
KEY RULE: All scores must be STRICTLY between 0 and 1.
  Never exactly 0.0 → use 0.001 minimum
  Never exactly 1.0 → use 0.999 maximum

This is required by OpenEnv validator Phase 2.
_clamp() enforces this on every single return value.
"""
import re
from typing import Dict, """
SYNAPSE — Graders (Final, Validator-Compliant)
================================================
KEY RULE: All scores must be STRICTLY between 0 and 1.
  Never exactly 0.0 → use 0.001 minimum
  Never exactly 1.0 → use 0.999 maximum

This is required by OpenEnv validator Phase 2.
_clamp() enforces this on every single return value.
"""
import re
from typing import Dict, List, Set
from models import Action, Reward


def _clamp(v: float) -> float:
    """
    Clamp to open interval (0.001, 0.999).
    OpenEnv validator REJECTS scores of exactly 0.0 or exactly 1.0.
    This function is called on EVERY reward returned.
    """
    return round(max(0.001, min(0.999, v)), 4)


def _keywords(text: str) -> Set[str]:
    """Extract meaningful keywords — underscores normalized, numbers removed."""
    STOP = {
        "to","the","a","an","and","or","for","of","in","from","with","at","by",
        "is","it","that","this","are","was","be","on","as","up","if","do","not",
        "before","after","then","than","all","any","some","very","just","now",
        "can","will","should","must","make","sure","check","verify","ensure",
        "immediately","properly","correctly","please","your","our","my",
    }
    text = text.lower().replace("_", " ").replace("=", " ").replace(".", " ")
    words = set(re.sub(r"[^a-z0-9\s]", " ", text).split())
    words = {w for w in words if not re.fullmatch(r"[0-9]+", w)}
    return words - STOP


def _match_score(agent_step: str, correct_step: str) -> float:
    """Jaccard similarity. Returns value in (0.001, 0.999)."""
    if agent_step.strip().lower() == correct_step.strip().lower():
        return 0.999  # perfect match — not 1.0
    ak = _keywords(agent_step)
    ck = _keywords(correct_step)
    if not ak or not ck:
        return 0.001  # no match — not 0.0
    j = len(ak & ck) / len(ak | ck)
    if j >= 0.7: return round(min(0.999, 0.90 + j * 0.09), 4)
    if j >= 0.5: return round(0.60 + j * 0.39, 4)
    if j >= 0.3: return round(j * 1.50, 4)
    if j > 0.0:  return round(j * 0.80, 4)
    return 0.001  # no keyword overlap — not 0.0


def _grade_steps(agent_steps: List[str], correct_steps: List[str], weight: float) -> tuple:
    """Grade list of steps. Returns (score_in_open_interval, feedback)."""
    if not agent_steps:
        return 0.001, "no steps provided"
    per_step = weight / len(correct_steps)
    total = 0.0
    hits = 0
    for cs in correct_steps:
        best = max((_match_score(a, cs) for a in agent_steps), default=0.001)
        total += per_step * best
        if best >= 0.7:
            hits += 1
    return _clamp(total), f"{hits}/{len(correct_steps)} steps matched"


# ── Task 1 — Signal Monitor (Easy) ───────────────────────────────────────────
def grade_task1(action: Action, gt: Dict) -> Reward:
    bd: Dict[str, float] = {}
    fb: List[str] = []

    # anomaly_detected (0.20)
    bd["anomaly_detected"] = 0.20 if action.anomaly_detected == gt["anomaly_detected"] else 0.001
    fb.append(f"{'✅' if action.anomaly_detected == gt['anomaly_detected'] else '❌'} anomaly_detected")

    # anomaly_type (0.50)
    bd["anomaly_type"] = 0.50 if action.anomaly_type == gt["anomaly_type"] else 0.001
    fb.append(f"{'✅' if action.anomaly_type == gt['anomaly_type'] else '❌'} type: {action.anomaly_type} vs {gt['anomaly_type']}")

    # severity (0.30) — partial for adjacent
    SEV = ["low", "medium", "high", "critical"]
    gs, es = action.severity, gt["severity"]
    if gs == es:
        bd["severity"] = 0.29  # not 0.30 to avoid potential 1.0
        fb.append(f"✅ severity: {gs}")
    elif gs in SEV and es in SEV and abs(SEV.index(gs) - SEV.index(es)) == 1:
        bd["severity"] = 0.15
        fb.append(f"⚠️ adjacent severity: {gs} vs {es}")
    else:
        bd["severity"] = 0.001
        fb.append(f"❌ severity: {gs} vs {es}")

    return Reward(score=_clamp(sum(bd.values())), breakdown=bd, feedback=" | ".join(fb))


# ── Task 2 — Root Cause Engine (Medium) ──────────────────────────────────────
def grade_task2(action: Action, gt: Dict) -> Reward:
    bd: Dict[str, float] = {}
    fb: List[str] = []

    # root_cause (0.60)
    bd["root_cause"] = 0.60 if action.root_cause == gt["root_cause"] else 0.001
    fb.append(f"{'✅' if action.root_cause == gt['root_cause'] else '❌'} root_cause: {action.root_cause} vs {gt['root_cause']}")

    # affected_components (0.39 — not 0.40 to avoid 1.0)
    correct_set = set(gt["affected_components"])
    agent_set = set(action.affected_components or [])
    if not agent_set:
        bd["components"] = 0.001
        fb.append("❌ no affected_components provided")
    else:
        hits = len(correct_set & agent_set)
        bd["components"] = round((hits / len(correct_set)) * 0.39, 4)
        wrong = len(agent_set - correct_set)
        if wrong > 0:
            penalty = round(min(wrong * 0.05, 0.10), 4)
            bd["component_penalty"] = -penalty
        fb.append(f"{'✅' if hits == len(correct_set) else '⚠️'} components: {hits}/{len(correct_set)}")

    return Reward(score=_clamp(sum(bd.values())), breakdown=bd, feedback=" | ".join(fb))


# ── Task 3 — Priority Classifier (Medium-Hard) ───────────────────────────────
def grade_task3(action: Action, gt: Dict) -> Reward:
    TEAMS = ["ml_team", "ai_team", "devops_team", "all_hands"]
    correct_team = gt["response_team"]
    agent_team = action.response_team

    if agent_team == correct_team:
        # 0.999 not 1.0 — validator rejects exactly 1.0
        return Reward(
            score=0.999,
            breakdown={"response_team": 0.999},
            feedback=f"✅ Correct team: {agent_team}",
        )
    elif agent_team in TEAMS and correct_team in TEAMS:
        dist = abs(TEAMS.index(agent_team) - TEAMS.index(correct_team))
        partial = 0.3 if dist == 1 else 0.001
        return Reward(
            score=partial,
            breakdown={"response_team": partial},
            feedback=f"{'⚠️ Adjacent' if dist == 1 else '❌ Wrong'} team: {agent_team} vs {correct_team}",
        )
    else:
        # 0.001 not 0.0 — validator rejects exactly 0.0
        return Reward(
            score=0.001,
            breakdown={"response_team": 0.001},
            feedback=f"❌ Wrong team: got {agent_team}, expected {correct_team}",
        )


# ── Task 4 — Remediation Planner (Hard) ──────────────────────────────────────
def grade_task4(action: Action, gt: Dict) -> Reward:
    bd: Dict[str, float] = {}
    fb: List[str] = []

    phases = [
        ("immediate_steps",    action.immediate_steps,    gt["immediate_steps"],    0.35),
        ("root_fix_steps",     action.root_fix_steps,     gt["root_fix_steps"],     0.39),  # 0.39 not 0.40
        ("verification_steps", action.verification_steps, gt["verification_steps"], 0.24),  # 0.24 not 0.25
    ]
    for pname, asteps, csteps, w in phases:
        if not asteps:
            bd[pname] = 0.001
            fb.append(f"❌ missing {pname}")
        else:
            s, msg = _grade_steps(asteps, csteps, w)
            bd[pname] = s
            fb.append(f"{pname}: {msg}")

    return Reward(score=_clamp(sum(bd.values())), breakdown=bd, feedback=" | ".join(fb))


# ── Task 5 — Post-Mortem Analyst (Very Hard) ─────────────────────────────────
def grade_task5(action: Action, gt: Dict) -> Reward:
    bd: Dict[str, float] = {}
    fb: List[str] = []

    if not action.postmortem:
        return Reward(
            score=0.001,
            breakdown={"postmortem": 0.001},
            feedback="❌ No postmortem provided",
        )
    pm = action.postmortem

    # root_cause (0.40)
    bd["root_cause"] = 0.40 if pm.get("root_cause") == gt["root_cause"] else 0.001
    fb.append(f"{'✅' if bd['root_cause'] > 0.01 else '❌'} root_cause: {pm.get('root_cause')} vs {gt['root_cause']}")

    # components (0.19 — not 0.20 to avoid 1.0)
    cc = set(gt["affected_components"])
    ac = set(pm.get("affected_components") or [])
    if cc and ac:
        overlap = len(cc & ac) / len(cc)
        bd["components"] = round(0.19 * overlap, 4)
        fb.append(f"components: {len(cc&ac)}/{len(cc)}")
    else:
        bd["components"] = 0.001
        fb.append("❌ missing components")

    # prevention_steps (0.19)
    cprev = list(gt["prevention_steps"])
    aprev = list(pm.get("prevention_steps") or [])
    if cprev and aprev:
        s, msg = _grade_steps(aprev, cprev, 0.19)
        bd["prevention"] = s
        fb.append(f"prevention: {msg}")
    else:
        bd["prevention"] = 0.001
        fb.append("❌ missing prevention_steps")

    # monitoring (0.19)
    cmon = list(gt["monitoring_additions"])
    amon = list(pm.get("monitoring_additions") or [])
    if cmon and amon:
        s, msg = _grade_steps(amon, cmon, 0.19)
        bd["monitoring"] = s
        fb.append(f"monitoring: {msg}")
    else:
        bd["monitoring"] = 0.001
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

BASELINE_TARGETS = {
    "task1": 0.85,
    "task2": 0.70,
    "task3": 0.80,
    "task4": 0.55,
    "task5": 0.45,
    "average": 0.67,
}
, Set
from models import Action, Reward


def _clamp(v: float) -> float:
    """
    Clamp to open interval (0.001, 0.999).
    OpenEnv validator REJECTS scores of exactly 0.0 or exactly 1.0.
    This function is called on EVERY reward returned.
    """
    return round(max(0.001, min(0.999, v)), 4)


def _keywords(text: str) -> Set[str]:
    """Extract meaningful keywords — underscores normalized, numbers removed."""
    STOP = {
        "to","the","a","an","and","or","for","of","in","from","with","at","by",
        "is","it","that","this","are","was","be","on","as","up","if","do","not",
        "before","after","then","than","all","any","some","very","just","now",
        "can","will","should","must","make","sure","check","verify","ensure",
        "immediately","properly","correctly","please","your","our","my",
    }
    text = text.lower().replace("_", " ").replace("=", " ").replace(".", " ")
    words = set(re.sub(r"[^a-z0-9\s]", " ", text).split())
    words = {w for w in words if not re.fullmatch(r"[0-9]+", w)}
    return words - STOP


def _match_score(agent_step: str, correct_step: str) -> float:
    """Jaccard similarity. Returns value in (0.001, 0.999)."""
    if agent_step.strip().lower() == correct_step.strip().lower():
        return 0.999  # perfect match — not 1.0
    ak = _keywords(agent_step)
    ck = _keywords(correct_step)
    if not ak or not ck:
        return 0.001  # no match — not 0.0
    j = len(ak & ck) / len(ak | ck)
    if j >= 0.7: return round(min(0.999, 0.90 + j * 0.09), 4)
    if j >= 0.5: return round(0.60 + j * 0.39, 4)
    if j >= 0.3: return round(j * 1.50, 4)
    if j > 0.0:  return round(j * 0.80, 4)
    return 0.001  # no keyword overlap — not 0.0


def _grade_steps(agent_steps: List[str], correct_steps: List[str], weight: float) -> tuple:
    """Grade list of steps. Returns (score_in_open_interval, feedback)."""
    if not agent_steps:
        return 0.001, "no steps provided"
    per_step = weight / len(correct_steps)
    total = 0.0
    hits = 0
    for cs in correct_steps:
        best = max((_match_score(a, cs) for a in agent_steps), default=0.001)
        total += per_step * best
        if best >= 0.7:
            hits += 1
    return _clamp(total), f"{hits}/{len(correct_steps)} steps matched"


# ── Task 1 — Signal Monitor (Easy) ───────────────────────────────────────────
def grade_task1(action: Action, gt: Dict) -> Reward:
    bd: Dict[str, float] = {}
    fb: List[str] = []

    # anomaly_detected (0.20)
    bd["anomaly_detected"] = 0.20 if action.anomaly_detected == gt["anomaly_detected"] else 0.001
    fb.append(f"{'✅' if action.anomaly_detected == gt['anomaly_detected'] else '❌'} anomaly_detected")

    # anomaly_type (0.50)
    bd["anomaly_type"] = 0.50 if action.anomaly_type == gt["anomaly_type"] else 0.001
    fb.append(f"{'✅' if action.anomaly_type == gt['anomaly_type'] else '❌'} type: {action.anomaly_type} vs {gt['anomaly_type']}")

    # severity (0.30) — partial for adjacent
    SEV = ["low", "medium", "high", "critical"]
    gs, es = action.severity, gt["severity"]
    if gs == es:
        bd["severity"] = 0.29  # not 0.30 to avoid potential 1.0
        fb.append(f"✅ severity: {gs}")
    elif gs in SEV and es in SEV and abs(SEV.index(gs) - SEV.index(es)) == 1:
        bd["severity"] = 0.15
        fb.append(f"⚠️ adjacent severity: {gs} vs {es}")
    else:
        bd["severity"] = 0.001
        fb.append(f"❌ severity: {gs} vs {es}")

    return Reward(score=_clamp(sum(bd.values())), breakdown=bd, feedback=" | ".join(fb))


# ── Task 2 — Root Cause Engine (Medium) ──────────────────────────────────────
def grade_task2(action: Action, gt: Dict) -> Reward:
    bd: Dict[str, float] = {}
    fb: List[str] = []

    # root_cause (0.60)
    bd["root_cause"] = 0.60 if action.root_cause == gt["root_cause"] else 0.001
    fb.append(f"{'✅' if action.root_cause == gt['root_cause'] else '❌'} root_cause: {action.root_cause} vs {gt['root_cause']}")

    # affected_components (0.39 — not 0.40 to avoid 1.0)
    correct_set = set(gt["affected_components"])
    agent_set = set(action.affected_components or [])
    if not agent_set:
        bd["components"] = 0.001
        fb.append("❌ no affected_components provided")
    else:
        hits = len(correct_set & agent_set)
        bd["components"] = round((hits / len(correct_set)) * 0.39, 4)
        wrong = len(agent_set - correct_set)
        if wrong > 0:
            penalty = round(min(wrong * 0.05, 0.10), 4)
            bd["component_penalty"] = -penalty
        fb.append(f"{'✅' if hits == len(correct_set) else '⚠️'} components: {hits}/{len(correct_set)}")

    return Reward(score=_clamp(sum(bd.values())), breakdown=bd, feedback=" | ".join(fb))


# ── Task 3 — Priority Classifier (Medium-Hard) ───────────────────────────────
def grade_task3(action: Action, gt: Dict) -> Reward:
    TEAMS = ["ml_team", "ai_team", "devops_team", "all_hands"]
    correct_team = gt["response_team"]
    agent_team = action.response_team

    if agent_team == correct_team:
        # 0.999 not 1.0 — validator rejects exactly 1.0
        return Reward(
            score=0.999,
            breakdown={"response_team": 0.999},
            feedback=f"✅ Correct team: {agent_team}",
        )
    elif agent_team in TEAMS and correct_team in TEAMS:
        dist = abs(TEAMS.index(agent_team) - TEAMS.index(correct_team))
        partial = 0.3 if dist == 1 else 0.001
        return Reward(
            score=partial,
            breakdown={"response_team": partial},
            feedback=f"{'⚠️ Adjacent' if dist == 1 else '❌ Wrong'} team: {agent_team} vs {correct_team}",
        )
    else:
        # 0.001 not 0.0 — validator rejects exactly 0.0
        return Reward(
            score=0.001,
            breakdown={"response_team": 0.001},
            feedback=f"❌ Wrong team: got {agent_team}, expected {correct_team}",
        )


# ── Task 4 — Remediation Planner (Hard) ──────────────────────────────────────
def grade_task4(action: Action, gt: Dict) -> Reward:
    bd: Dict[str, float] = {}
    fb: List[str] = []

    phases = [
        ("immediate_steps",    action.immediate_steps,    gt["immediate_steps"],    0.35),
        ("root_fix_steps",     action.root_fix_steps,     gt["root_fix_steps"],     0.39),  # 0.39 not 0.40
        ("verification_steps", action.verification_steps, gt["verification_steps"], 0.24),  # 0.24 not 0.25
    ]
    for pname, asteps, csteps, w in phases:
        if not asteps:
            bd[pname] = 0.001
            fb.append(f"❌ missing {pname}")
        else:
            s, msg = _grade_steps(asteps, csteps, w)
            bd[pname] = s
            fb.append(f"{pname}: {msg}")

    return Reward(score=_clamp(sum(bd.values())), breakdown=bd, feedback=" | ".join(fb))


# ── Task 5 — Post-Mortem Analyst (Very Hard) ─────────────────────────────────
def grade_task5(action: Action, gt: Dict) -> Reward:
    bd: Dict[str, float] = {}
    fb: List[str] = []

    if not action.postmortem:
        return Reward(
            score=0.001,
            breakdown={"postmortem": 0.001},
            feedback="❌ No postmortem provided",
        )
    pm = action.postmortem

    # root_cause (0.40)
    bd["root_cause"] = 0.40 if pm.get("root_cause") == gt["root_cause"] else 0.001
    fb.append(f"{'✅' if bd['root_cause'] > 0.01 else '❌'} root_cause: {pm.get('root_cause')} vs {gt['root_cause']}")

    # components (0.19 — not 0.20 to avoid 1.0)
    cc = set(gt["affected_components"])
    ac = set(pm.get("affected_components") or [])
    if cc and ac:
        overlap = len(cc & ac) / len(cc)
        bd["components"] = round(0.19 * overlap, 4)
        fb.append(f"components: {len(cc&ac)}/{len(cc)}")
    else:
        bd["components"] = 0.001
        fb.append("❌ missing components")

    # prevention_steps (0.19)
    cprev = list(gt["prevention_steps"])
    aprev = list(pm.get("prevention_steps") or [])
    if cprev and aprev:
        s, msg = _grade_steps(aprev, cprev, 0.19)
        bd["prevention"] = s
        fb.append(f"prevention: {msg}")
    else:
        bd["prevention"] = 0.001
        fb.append("❌ missing prevention_steps")

    # monitoring (0.19)
    cmon = list(gt["monitoring_additions"])
    amon = list(pm.get("monitoring_additions") or [])
    if cmon and amon:
        s, msg = _grade_steps(amon, cmon, 0.19)
        bd["monitoring"] = s
        fb.append(f"monitoring: {msg}")
    else:
        bd["monitoring"] = 0.001
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

BASELINE_TARGETS = {
    "task1": 0.85,
    "task2": 0.70,
    "task3": 0.80,
    "task4": 0.55,
    "task5": 0.45,
    "average": 0.67,
}
