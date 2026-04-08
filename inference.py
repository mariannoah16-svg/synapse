"""
SYNAPSE — Inference Script (Final)
====================================
Uses FREE HuggingFace Inference Router by default.
No paid OpenAI API key needed.

Required env vars:
  HF_TOKEN     — Free HuggingFace token (https://huggingface.co/settings/tokens)
  API_BASE_URL — Default: https://api-inference.huggingface.co/v1 (FREE)
  MODEL_NAME   — Default: meta-llama/Llama-3.1-8B-Instruct (FREE on HF)
  ENV_URL      — Default: http://localhost:7860

Usage:
  python inference.py                    # pretty output
  python inference.py --mode api         # JSON for /baseline endpoint

Fallback:
  If no HF_TOKEN set → rule-based agent scores ~0.35
  Proves environment works without any API key.
"""

import argparse, json, os, re, sys, time
import requests
from openai import OpenAI

# ── Config — HuggingFace FREE by default ─────────────────────────────────────
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://api-inference.huggingface.co/v1"   # FREE HF router
)
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "meta-llama/Llama-3.1-8B-Instruct"         # FREE on HF
)
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
ENV_URL  = os.getenv("ENV_URL", "http://localhost:7860")
TASK_IDS = ["task1", "task2", "task3", "task4", "task5"]

# OpenAI-compatible client — works with HF, OpenAI, Groq, etc.
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "no-key",
)

SYSTEM = """You are a world-class senior ML engineer specializing in PyTorch production systems.
Diagnose AI model failures from the provided incident data.
Respond with ONLY valid JSON — no explanation, no markdown."""

# ── Rule-based fallback (no API key needed) ───────────────────────────────────
def _rule_based(task_id: str, obs: dict) -> dict:
    """Heuristic agent — scores ~0.35 average without any API key."""
    logs    = obs.get("training_logs", [])
    metrics = obs.get("production_metrics", {})
    errors  = obs.get("error_traces", [])
    cfg     = obs.get("training_config", {})
    summary = obs.get("alert_summary", "").lower()

    # Detect anomaly type
    atype = "accuracy_drop"
    for e in errors:
        m = e.get("message","").lower()
        if "nan" in m:              atype = "nan_loss";          break
        if "out of memory" in m:    atype = "gpu_oom";           break
        if "gradient" in m:         atype = "gradient_explosion";break
        if "sequence length" in m:  atype = "hallucination_spike";break
    for l in logs:
        if l.get("error"):
            if "nan"    in str(l["error"]).lower(): atype = "nan_loss";  break
            if "memory" in str(l["error"]).lower(): atype = "gpu_oom";   break
    if "hallucin" in summary:       atype = "hallucination_spike"
    if "drift"    in summary:       atype = "data_drift"
    if "injection"in summary:       atype = "hallucination_spike"

    # Severity
    er = metrics.get("error_rate", 0)
    sev = ("critical" if er >= 0.8 or atype in ["nan_loss","gpu_oom","gradient_explosion"]
           else "high" if er >= 0.3
           else "medium" if er >= 0.1
           else "low")

    # Root cause
    lr      = cfg.get("learning_rate", 0.001)
    opt     = cfg.get("optimizer","Adam")
    typical = {"Adam":0.001,"AdamW":0.0001,"SGD":0.01,"RMSprop":0.001}.get(opt, 0.001)
    rc = "bad_deployment"
    if atype == "nan_loss" and lr > typical * 5:    rc = "learning_rate_too_high"
    elif atype == "gpu_oom":                        rc = "memory_leak"
    elif atype == "gradient_explosion":             rc = "gradient_explosion"
    elif atype == "data_drift":                     rc = "data_drift"
    elif atype == "hallucination_spike":
        rc = "context_overflow" if any("sequence" in str(e.get("message","")).lower() for e in errors) else "prompt_injection"
    elif any("overfitting" in str(l.get("warning","")).lower() for l in logs): rc = "overfitting"
    elif lr < typical * 0.01:                       rc = "learning_rate_too_low"

    # Team
    team = ("ai_team"     if rc in ["prompt_injection","context_overflow"]
            else "devops_team" if rc == "bad_deployment"
            else "ml_team")

    COMP = {
        "learning_rate_too_high":["optimizer","loss_function"],
        "learning_rate_too_low": ["optimizer"],
        "memory_leak":           ["dataloader","model"],
        "gradient_explosion":    ["optimizer","model"],
        "overfitting":           ["model","loss_function"],
        "data_drift":            ["model"],
        "bad_deployment":        ["model"],
        "context_overflow":      ["model"],
        "prompt_injection":      ["model"],
        "class_imbalance":       ["model","loss_function"],
    }
    comp = COMP.get(rc, ["model"])

    if task_id == "task1":
        return {"anomaly_detected": True, "anomaly_type": atype, "severity": sev}
    elif task_id == "task2":
        return {"root_cause": rc, "affected_components": comp}
    elif task_id == "task3":
        return {"priority_ranking": ["primary-model-prod"], "response_team": team}
    elif task_id == "task4":
        FIXES = {
            "learning_rate_too_high": {
                "immediate_steps":    ["stop training", "revert learning rate"],
                "root_fix_steps":     ["add gradient clipping", "use learning rate scheduler", "restart training"],
                "verification_steps": ["verify loss decreasing", "verify gradient norm below 10"],
            },
            "memory_leak": {
                "immediate_steps":    ["stop training", "reduce batch size"],
                "root_fix_steps":     ["enable gradient checkpointing", "use mixed precision fp16", "restart training"],
                "verification_steps": ["verify gpu memory below 80 percent", "verify training completes"],
            },
            "overfitting": {
                "immediate_steps":    ["rollback to previous model", "disable current deployment"],
                "root_fix_steps":     ["add dropout 0.3", "add weight decay 0.01", "retrain with early stopping"],
                "verification_steps": ["verify train val gap below 5 percent", "verify accuracy above 80 percent"],
            },
        }
        default = {
            "immediate_steps":    ["stop current process", "assess the damage"],
            "root_fix_steps":     ["identify root cause", "apply targeted fix", "test the fix"],
            "verification_steps": ["verify metrics improved", "verify no regression"],
        }
        return FIXES.get(rc, default)
    else:  # task5
        return {"postmortem": {
            "root_cause": rc,
            "affected_components": comp,
            "prevention_steps": ["add monitoring alerts", "implement automated testing"],
            "monitoring_additions": ["accuracy degradation alert", "error rate monitor"],
        }}


# ── JSON extraction ───────────────────────────────────────────────────────────
def _extract_json(text: str) -> dict:
    """Robustly extract the last valid JSON object from LLM output."""
    # Try full text first
    try:
        clean = text.replace("```json","").replace("```","").strip()
        return json.loads(clean)
    except Exception:
        pass
    # Find all JSON-like blocks and try each from last to first
    blocks = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', text, re.DOTALL)
    for block in reversed(blocks):
        try:
            return json.loads(block)
        except Exception:
            pass
    return {}


# ── LLM agent ────────────────────────────────────────────────────────────────
def call_llm(task_id: str, obs: dict) -> dict:
    """Call LLM via HuggingFace router (free) or OpenAI-compatible API."""
    if not HF_TOKEN:
        return _rule_based(task_id, obs)

    cfg     = obs.get("training_config", {})
    metrics = obs.get("production_metrics", {})
    logs    = obs.get("training_logs", [])
    errors  = obs.get("error_traces", [])
    deploys = obs.get("deployment_history", [])
    hint    = obs.get("hint", "")

    logs_txt = "\n".join(
        f"  Epoch {l.get('epoch')}/{l.get('total_epochs')} "
        f"loss={l.get('train_loss')} acc={l.get('train_acc')}% lr={l.get('learning_rate')}"
        f"{' | WARN:'+l['warning'] if l.get('warning') else ''}"
        f"{' | ERR:'+l['error']   if l.get('error')   else ''}"
        for l in logs) or "  none"

    err_txt = "\n".join(
        f"  [{e.get('error_type')}] {e.get('pytorch_component')}: {e.get('message')}"
        for e in errors) or "  none"

    dep_txt = "\n".join(
        f"  [{d.get('event_type')}] {d.get('version')}: {d.get('description')}"
        for d in deploys) or "  none"

    context = f"""INCIDENT: {obs.get('alert_summary')}
{f'HINT: {hint}' if hint else ''}
MODEL: {cfg.get('architecture')} | {cfg.get('optimizer')} lr={cfg.get('learning_rate')} batch={cfg.get('batch_size')} clip={cfg.get('gradient_clip')}
METRICS: accuracy={metrics.get('accuracy')} error_rate={metrics.get('error_rate')} gpu={metrics.get('gpu_memory_used_gb')}/{metrics.get('gpu_memory_total_gb')}GB drift={metrics.get('drift_score')}
TRAINING LOGS:
{logs_txt}
ERRORS:
{err_txt}
DEPLOYMENTS:
{dep_txt}"""

    SCHEMAS = {
        "task1": '{"anomaly_detected":true,"anomaly_type":"nan_loss|gpu_oom|accuracy_drop|data_drift|latency_spike|hallucination_spike|gradient_explosion|memory_leak|loss_spike|none","severity":"low|medium|high|critical"}',
        "task2": '{"root_cause":"learning_rate_too_high|learning_rate_too_low|overfitting|underfitting|data_drift|quantization_error|prompt_injection|context_overflow|bad_deployment|class_imbalance|gradient_explosion|memory_leak","affected_components":["component1"]}',
        "task3": '{"priority_ranking":["model-name"],"response_team":"ml_team|ai_team|devops_team|all_hands"}',
        "task4": '{"immediate_steps":["step1","step2"],"root_fix_steps":["step1","step2"],"verification_steps":["step1"]}',
        "task5": '{"postmortem":{"root_cause":"str","affected_components":["comp"],"prevention_steps":["step"],"monitoring_additions":["monitor"]}}',
    }

    prompt = f"{context}\n\nTASK {task_id}: Analyze above then respond ONLY with JSON:\n{SCHEMAS[task_id]}"

    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=512,
        )
        raw    = r.choices[0].message.content.strip()
        result = _extract_json(raw)
        return result if result else _rule_based(task_id, obs)
    except Exception as e:
        print(f"  LLM error ({task_id}): {e}", file=sys.stderr)
        return _rule_based(task_id, obs)


# ── Task runner ───────────────────────────────────────────────────────────────
def run_task(task_id: str, seed: int = 42) -> tuple:
    t0 = time.time()
    try:
        r = requests.post(f"{ENV_URL}/reset",
                          json={"task_id": task_id, "seed": seed}, timeout=30)
        r.raise_for_status()
        obs = r.json()["observation"]
    except Exception as e:
        return 0.0, f"reset failed: {e}", time.time() - t0

    action = call_llm(task_id, obs)

    try:
        r = requests.post(f"{ENV_URL}/step",
                          json={"action": action}, timeout=30)
        r.raise_for_status()
        res = r.json()
        return res["reward"]["score"], res["reward"]["feedback"], time.time() - t0
    except Exception as e:
        return 0.0, f"step failed: {e}", time.time() - t0


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SYNAPSE Baseline Inference")
    parser.add_argument("--mode", choices=["pretty", "api"], default="pretty")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    NAMES = {
        "task1": "Signal Monitor     (Easy)       ",
        "task2": "Root Cause Engine  (Medium)     ",
        "task3": "Priority Classifier(Med-Hard)   ",
        "task4": "Remediation Planner(Hard)       ",
        "task5": "Post-Mortem Analyst(Very Hard)  ",
    }

    results   = {}
    total_t   = 0.0
    agent_mode = "LLM: " + MODEL_NAME if HF_TOKEN else "rule-based fallback"

    if args.mode == "pretty":
        print(f"\n  SYNAPSE — Baseline Inference ({agent_mode})")
        print("  " + "="*55)
        
        # --- REQUIRED BY SCALER HACKATHON ---
        print("[START]") 

    for tid in TASK_IDS:
        score, feedback, elapsed = run_task(tid, seed=args.seed)
        results[tid] = {
            "score":           score,
            "feedback":        feedback,
            "elapsed_seconds": round(elapsed, 2),
        }
        total_t += elapsed
        
        if args.mode == "pretty":
            print(f"  {NAMES[tid]}: {score:.4f}  ({elapsed:.1f}s)")
            
            # --- REQUIRED BY SCALER HACKATHON ---
            print(f"[STEP] task_id={tid} score={score}")

    avg = round(sum(r["score"] for r in results.values()) / len(results), 4)
    results["average"]               = avg
    results["total_elapsed_seconds"] = round(total_t, 2)
    results["model_used"]            = MODEL_NAME if HF_TOKEN else "rule_based_fallback"
    results["seed"]                  = args.seed

    if args.mode == "api":
        # Last line must be parseable JSON — required by /baseline endpoint
        print(json.dumps(results))
    else:
        print(f"\n  Average:   {avg:.4f}  (target: ~0.67)")
        print(f"  Runtime:   {total_t:.1f}s  (limit: 1200s)")
        print(f"  Agent:     {agent_mode}")
        if not HF_TOKEN:
            print(f"\n  ⚠️  Set HF_TOKEN env var for LLM-based inference.")
            print(f"  Get free token: https://huggingface.co/settings/tokens")
        print("  " + "="*55 + "\n")
        
        # --- REQUIRED BY SCALER HACKATHON ---
        print("[END]")


if __name__ == "__main__":
    main()
