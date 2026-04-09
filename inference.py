"""
SYNAPSE — Inference Script (Final, Validator-Compliant)
=========================================================
Emits [START], [STEP], [END] structured stdout logs.
Required by OpenEnv validator Phase 2.

Uses FREE HuggingFace Inference Router by default.
Falls back to rule-based agent when no token set.
"""

import argparse, json, os, re, sys, time
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")
TASK_IDS     = ["task1", "task2", "task3", "task4", "task5"]

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-key")

SYSTEM = """You are a senior ML engineer. Diagnose AI model failures.
Respond with ONLY valid JSON — no explanation, no markdown."""

# ── Rule-based fallback ───────────────────────────────────────────────────────
def _rule_based(task_id: str, obs: dict) -> dict:
    """Returns valid action even with no API key. Scores ~0.35."""
    logs    = obs.get("training_logs", [])
    metrics = obs.get("production_metrics", {})
    errors  = obs.get("error_traces", [])
    cfg     = obs.get("training_config", {})
    summary = obs.get("alert_summary", "").lower()

    atype = "accuracy_drop"
    for e in errors:
        m = e.get("message", "").lower()
        if "nan" in m:             atype = "nan_loss";          break
        if "out of memory" in m:   atype = "gpu_oom";           break
        if "gradient" in m:        atype = "gradient_explosion"; break
        if "sequence length" in m: atype = "hallucination_spike";break
    for l in logs:
        err = str(l.get("error", "")).lower()
        if "nan"    in err: atype = "nan_loss";  break
        if "memory" in err: atype = "gpu_oom";   break
    if "hallucin"  in summary: atype = "hallucination_spike"
    if "drift"     in summary: atype = "data_drift"

    er  = metrics.get("error_rate", 0)
    sev = ("critical" if er >= 0.8 or atype in ["nan_loss","gpu_oom","gradient_explosion"]
           else "high"   if er >= 0.3
           else "medium" if er >= 0.1
           else "low")

    lr      = cfg.get("learning_rate", 0.001)
    opt     = cfg.get("optimizer", "Adam")
    typical = {"Adam":0.001,"AdamW":0.0001,"SGD":0.01,"RMSprop":0.001}.get(opt, 0.001)
    rc = "bad_deployment"
    if atype == "nan_loss" and lr > typical * 5:   rc = "learning_rate_too_high"
    elif atype == "gpu_oom":                        rc = "memory_leak"
    elif atype == "gradient_explosion":             rc = "gradient_explosion"
    elif atype == "data_drift":                     rc = "data_drift"
    elif atype == "hallucination_spike":
        rc = "context_overflow" if any("sequence" in str(e.get("message","")).lower() for e in errors) else "prompt_injection"
    elif any("overfitting" in str(l.get("warning","")).lower() for l in logs): rc = "overfitting"
    elif lr < typical * 0.01: rc = "learning_rate_too_low"

    team = ("ai_team"     if rc in ["prompt_injection","context_overflow"]
            else "devops_team" if rc == "bad_deployment"
            else "ml_team")

    COMP = {
        "learning_rate_too_high": ["optimizer","loss_function"],
        "learning_rate_too_low":  ["optimizer"],
        "memory_leak":            ["dataloader","model"],
        "gradient_explosion":     ["optimizer","model"],
        "overfitting":            ["model","loss_function"],
        "data_drift":             ["model"],
        "bad_deployment":         ["model"],
        "context_overflow":       ["model"],
        "prompt_injection":       ["model"],
        "class_imbalance":        ["model","loss_function"],
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
                "verification_steps": ["verify gpu memory stable", "verify training completes"],
            },
            "overfitting": {
                "immediate_steps":    ["rollback model", "disable current deployment"],
                "root_fix_steps":     ["add dropout 0.3", "add weight decay 0.01", "retrain with early stopping"],
                "verification_steps": ["verify train val gap below 5 percent", "verify accuracy above 80 percent"],
            },
        }
        default = {
            "immediate_steps":    ["stop current process", "assess damage"],
            "root_fix_steps":     ["identify root cause", "apply fix", "test fix"],
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
    try:
        clean = text.replace("```json","").replace("```","").strip()
        return json.loads(clean)
    except Exception:
        pass
    blocks = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', text, re.DOTALL)
    for block in reversed(blocks):
        try:
            return json.loads(block)
        except Exception:
            pass
    return {}


# ── LLM call ─────────────────────────────────────────────────────────────────
def call_llm(task_id: str, obs: dict) -> dict:
    if not HF_TOKEN:
        return _rule_based(task_id, obs)
    cfg     = obs.get("training_config", {})
    metrics = obs.get("production_metrics", {})
    logs    = obs.get("training_logs", [])
    errors  = obs.get("error_traces", [])
    deploys = obs.get("deployment_history", [])

    logs_txt = "\n".join(
        f"Epoch {l.get('epoch')}: loss={l.get('train_loss')} acc={l.get('train_acc')}%"
        f"{' WARN:'+l['warning'] if l.get('warning') else ''}"
        f"{' ERR:'+l['error'] if l.get('error') else ''}"
        for l in logs) or "none"
    err_txt = "\n".join(f"[{e.get('error_type')}] {e.get('message')}" for e in errors) or "none"
    dep_txt = "\n".join(f"[{d.get('event_type')}] {d.get('description')}" for d in deploys) or "none"

    context = (
        f"INCIDENT: {obs.get('alert_summary')}\n"
        f"MODEL: {cfg.get('architecture')} lr={cfg.get('learning_rate')} opt={cfg.get('optimizer')}\n"
        f"METRICS: accuracy={metrics.get('accuracy')} error_rate={metrics.get('error_rate')} "
        f"gpu={metrics.get('gpu_memory_used_gb')}/{metrics.get('gpu_memory_total_gb')}GB\n"
        f"LOGS:\n{logs_txt}\nERRORS:\n{err_txt}\nDEPLOYMENTS:\n{dep_txt}"
    )
    SCHEMAS = {
        "task1": '{"anomaly_detected":true,"anomaly_type":"nan_loss|gpu_oom|accuracy_drop|data_drift|latency_spike|hallucination_spike|gradient_explosion|memory_leak|loss_spike|none","severity":"low|medium|high|critical"}',
        "task2": '{"root_cause":"learning_rate_too_high|learning_rate_too_low|overfitting|underfitting|data_drift|quantization_error|prompt_injection|context_overflow|bad_deployment|class_imbalance|gradient_explosion|memory_leak","affected_components":["component"]}',
        "task3": '{"priority_ranking":["model-name"],"response_team":"ml_team|ai_team|devops_team|all_hands"}',
        "task4": '{"immediate_steps":["step1"],"root_fix_steps":["step1"],"verification_steps":["step1"]}',
        "task5": '{"postmortem":{"root_cause":"str","affected_components":["comp"],"prevention_steps":["step"],"monitoring_additions":["monitor"]}}',
    }
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": f"{context}\n\nRespond ONLY with JSON:\n{SCHEMAS[task_id]}"},
            ],
            temperature=0, max_tokens=512,
        )
        raw    = r.choices[0].message.content.strip()
        result = _extract_json(raw)
        return result if result else _rule_based(task_id, obs)
    except Exception as e:
        print(f"[WARN] LLM error ({task_id}): {e}", file=sys.stderr)
        return _rule_based(task_id, obs)


# ── Run one task ──────────────────────────────────────────────────────────────
def run_task(task_id: str, seed: int) -> dict:
    """Run one full episode. Returns result dict with score."""
    t0 = time.time()

    # Reset
    try:
        r = requests.post(f"{ENV_URL}/reset",
                          json={"task_id": task_id, "seed": seed}, timeout=30)
        r.raise_for_status()
        obs = r.json()["observation"]
    except Exception as e:
        return {"task_id": task_id, "score": 0.0001, "error": str(e),
                "elapsed": round(time.time()-t0, 2)}

    # Get action
    action = call_llm(task_id, obs)

    # Step
    try:
        r = requests.post(f"{ENV_URL}/step",
                          json={"action": action}, timeout=30)
        r.raise_for_status()
        res = r.json()
        return {
            "task_id":  task_id,
            "score":    res["reward"]["score"],
            "feedback": res["reward"]["feedback"],
            "elapsed":  round(time.time()-t0, 2),
        }
    except Exception as e:
        return {"task_id": task_id, "score": 0.0001, "error": str(e),
                "elapsed": round(time.time()-t0, 2)}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pretty", "api"], default="pretty")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed    = args.seed
    results = {}

    # ── Run each task with Exact Literal [START] and [STEP] logs ───────────────
    for task_id in TASK_IDS:
        # REQUIRED BY VALIDATOR: Exact string format, flushed immediately
        print(f"[START] task={task_id}", flush=True)
        
        result = run_task(task_id, seed=seed)
        score  = result["score"]
        results[task_id] = score

        # REQUIRED BY VALIDATOR: Exact string format, flushed immediately
        print(f"[STEP] step=1 reward={score}", flush=True)
        print(f"[END] task={task_id} score={score} steps=1", flush=True)

    # ── Final JSON line for /baseline endpoint (MUST REMAIN JSON) ─────────────
    avg = round(sum(results.values()) / len(results), 4)
    final = {**results, "average": avg, "seed": seed,
             "model_used": MODEL_NAME if HF_TOKEN else "rule_based_fallback"}
    
    if args.mode == "api":
        print(json.dumps(final), flush=True)
    else:
        print(f"\n  Average: {avg:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
