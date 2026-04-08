"""
SYNAPSE — Demo Script
=======================
Interactive demonstration of the SYNAPSE environment.

This script shows how any AI agent can interact with SYNAPSE
using the standard OpenEnv step/reset/state API.

USAGE:
  python demo.py                    # Run full demo (all 5 tasks)
  python demo.py --task task1       # Demo specific task only
  python demo.py --url http://...   # Use custom server URL

No API key needed — demo uses pre-defined example actions.
"""

import argparse
import json
import requests

ENV_URL = "http://localhost:7860"


def print_observation(obs: dict, task_id: str):
    """Pretty-print the observation for demo purposes."""
    cfg = obs.get("training_config", {})
    metrics = obs.get("production_metrics", {})

    print(f"\n  📋 INCIDENT ALERT:")
    print(f"     {obs.get('alert_summary')}")
    if obs.get("hint"):
        print(f"  💡 HINT: {obs['hint']}")
    print(f"\n  🔧 MODEL: {cfg.get('architecture')} | LR: {cfg.get('learning_rate')} | Batch: {cfg.get('batch_size')}")
    print(f"  📊 PRODUCTION: Accuracy={metrics.get('accuracy'):.0%} | Error Rate={metrics.get('error_rate'):.0%} | Drift={metrics.get('drift_score'):.2f}")

    if obs.get("error_traces"):
        err = obs["error_traces"][0]
        print(f"  ❌ ERROR: [{err['error_type']}] {err['message'][:100]}...")

    logs = obs.get("training_logs", [])
    if logs:
        last_log = logs[-1]
        print(f"  📈 LAST EPOCH: loss={last_log.get('train_loss')} | acc={last_log.get('train_acc')}%", end="")
        if last_log.get("error"):
            print(f" | ❌ {last_log['error'][:60]}")
        elif last_log.get("warning"):
            print(f" | ⚠️  {last_log['warning'][:60]}")
        else:
            print()


def demo_task(task_id: str, example_action: dict, server_url: str):
    """Run a complete episode demo for one task."""
    task_info = {
        "task1": ("Signal Monitor",      "Easy",       "Detect anomaly type and severity"),
        "task2": ("Root Cause Engine",   "Medium",     "Identify root cause and components"),
        "task3": ("Priority Classifier", "Med-Hard",   "Assign priority and response team"),
        "task4": ("Remediation Planner", "Hard",       "Plan 3-phase remediation"),
        "task5": ("Post-Mortem Analyst", "Very Hard",  "Write structured post-mortem"),
    }
    name, difficulty, desc = task_info[task_id]

    print(f"\n{'='*65}")
    print(f"  🧠 SYNAPSE Demo — {name} ({difficulty})")
    print(f"  {desc}")
    print(f"{'='*65}")

    # Reset environment
    try:
        resp = requests.post(f"{server_url}/reset", json={"task_id": task_id, "seed": 42}, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ❌ Could not connect to SYNAPSE server at {server_url}")
        print(f"     Error: {e}")
        print(f"     Start the server first: uvicorn server.app:app --port 7860")
        return

    observation = resp.json()["observation"]
    print_observation(observation, task_id)

    # Show the action we're submitting
    print(f"\n  🤖 AGENT ACTION:")
    print(f"     {json.dumps(example_action, indent=6)}")

    # Submit action
    resp = requests.post(f"{server_url}/step", json={"action": example_action}, timeout=10)
    resp.raise_for_status()
    result = resp.json()

    reward = result["reward"]
    score = reward["score"]
    feedback = reward["feedback"]

    # Show result
    score_bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
    print(f"\n  📊 GRADER RESULT:")
    print(f"     Score: {score:.4f} [{score_bar}]")
    print(f"     {feedback}")
    print(f"\n  Breakdown: {json.dumps(reward['breakdown'], indent=6)}")


def main():
    parser = argparse.ArgumentParser(description="SYNAPSE Environment Demo")
    parser.add_argument("--task", choices=["task1","task2","task3","task4","task5","all"], default="all")
    parser.add_argument("--url", default=ENV_URL, help="SYNAPSE server URL")
    args = parser.parse_args()

    server_url = args.url.rstrip("/")

    print("\n" + "="*65)
    print("  🧠 SYNAPSE — AI Model Incident Response Environment")
    print("  OpenEnv Hackathon 2026 | Team: Hive Mind")
    print("="*65)
    print(f"\n  Connecting to: {server_url}")

    # Verify server is running
    try:
        health = requests.get(f"{server_url}/health", timeout=5)
        health.raise_for_status()
        print(f"  ✅ Server status: {health.json()['status']}")
    except Exception as e:
        print(f"  ❌ Server not reachable: {e}")
        print(f"     Run: uvicorn server.app:app --host 0.0.0.0 --port 7860")
        return

    # Example actions for each task (showing what a well-performing agent does)
    example_actions = {
        "task1": {
            "anomaly_detected": True,
            "anomaly_type": "nan_loss",
            "severity": "critical",
        },
        "task2": {
            "root_cause": "learning_rate_too_high",
            "affected_components": ["optimizer", "loss_function"],
        },
        "task3": {
            "priority_ranking": ["resnet50-prod"],
            "response_team": "ml_team",
        },
        "task4": {
            "immediate_steps": ["stop training immediately", "revert learning_rate to 0.001"],
            "root_fix_steps": ["add gradient_clip max_norm=1.0", "use lr_scheduler with warmup", "restart from last valid checkpoint"],
            "verification_steps": ["verify loss decreasing for 3 epochs", "verify gradient norm below 10.0"],
        },
        "task5": {
            "postmortem": {
                "root_cause": "learning_rate_too_high",
                "affected_components": ["optimizer", "loss_function"],
                "prevention_steps": ["add_lr_scheduler", "add_gradient_clipping"],
                "monitoring_additions": ["nan_loss_alert", "gradient_norm_monitor"],
            }
        },
    }

    if args.task == "all":
        for task_id in ["task1", "task2", "task3", "task4", "task5"]:
            demo_task(task_id, example_actions[task_id], server_url)
    else:
        demo_task(args.task, example_actions[args.task], server_url)

    print(f"\n{'='*65}")
    print("  ✅ Demo complete!")
    print(f"  📖 Full API docs: {server_url}/docs")
    print(f"  📊 All tasks: {server_url}/tasks")
    print(f"  📈 Baseline scores: {server_url}/baseline")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()