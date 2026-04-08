---
title: SYNAPSE
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
  - pytorch
  - mlops
  - incident-response
  - reinforcement-learning
license: mit
---

# 🧠 SYNAPSE v5
### Systematic Neural Analysis and Production Supervision Environment

> **OpenEnv Hackathon 2026 | Meta × PyTorch × Scaler | Team: Hive Mind**

The world's first OpenEnv environment with **multi-turn investigation**, **procedural generation**, and **curriculum learning** — where AI agents learn to diagnose and fix AI model failures in production.

---

## 🎯 What SYNAPSE Does

```
Every AI company — Meta, OpenAI, Google — operates AI models in production 24/7.
When these models fail:

  Detection  → Hours    (manual monitoring)
  Diagnosis  → Days     (senior engineers debugging)
  Cost       → Millions per hour of downtime

SYNAPSE trains AI agents to handle this in minutes, automatically.
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     SYNAPSE v5                          │
│                                                         │
│  ScenarioGenerator    Environment       FastAPI Server  │
│  ───────────────── → ─────────────── → ──────────────  │
│  12 failure modes     reset()           /reset          │
│  10 architectures     step()            /step           │
│  = unlimited          state()           /state          │
│    scenarios                            /tasks          │
│                     CurriculumManager   /grader         │
│  ───────────────── → ─────────────── → /baseline       │
│  Easy→Medium→Hard     auto difficulty   /analytics      │
│  auto progression     scaling           /curriculum     │
│                                         /generate       │
│                       Graders           /train          │
│                     ─────────────────   /web  /docs     │
│                     Flexible keyword    /  (dashboard)  │
│                     Jaccard matching                    │
└─────────────────────────────────────────────────────────┘
```

---

## 🔍 Investigation Mode (Multi-Turn)

```
PHASE 1 — TRIAGE (weight: 0.30)
  Agent sees:  Alert + Production Metrics only
  Agent does:  Detect anomaly_type + severity
  Reward:      0.0 – 0.30
  Unlocks:     Training logs + Error traces

PHASE 2 — DIAGNOSIS (weight: 0.30)
  Agent sees:  + Training logs + Error traces
  Agent does:  Identify root_cause + components
  Reward:      cumulative up to 0.60
  Unlocks:     Full deployment history

PHASE 3 — RESOLUTION (weight: 0.40)
  Agent sees:  Complete context (everything)
  Agent does:  Plan immediate + root fix + verification
  Reward:      cumulative up to 1.0
  Done:        True
```

---

## 📦 The 5 Standard Tasks

| Task | Difficulty | Description | Baseline |
|------|-----------|-------------|---------|
| task1 — Signal Monitor | Easy | Detect anomaly + severity | ~0.85 |
| task2 — Root Cause Engine | Medium | Find root cause + components | ~0.70 |
| task3 — Priority Classifier | Medium-Hard | Assign response team | ~0.80 |
| task4 — Remediation Planner | Hard | 3-phase fix plan | ~0.55 |
| task5 — Post-Mortem Analyst | Very Hard | Structured post-mortem | ~0.45 |
| investigation — Full | Progressive | Multi-turn 3-phase | ~0.65 |

---

## 🎭 12 Failure Modes

| Failure | Difficulty | Root Cause |
|---------|-----------|-----------|
| NaN Loss | Easy | Learning rate too high |
| GPU OOM | Easy | Batch size too large |
| Gradient Explosion | Easy | No gradient clipping |
| Overfitting | Medium | No dropout/weight_decay |
| Underfitting | Medium | Learning rate too low |
| Bad Deployment | Medium | Wrong model weights |
| Memory Leak | Medium | GPU augmentation in DataLoader |
| Data Drift | Hard | Distribution shift |
| Class Imbalance | Hard | 96% dominant class |
| Context Overflow | Hard | Token limit exceeded |
| Quantization Error | Hard | INT8 without QAT |
| Prompt Injection | Hard | Safety filter disabled |

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run server
uv run server
# OR: python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

# Test (no API key needed)
python inference.py

# With free HuggingFace token (better scores)
export HF_TOKEN=hf_your_token_here
python inference.py
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Visual dashboard |
| GET | `/web` | OpenEnv web interface |
| GET | `/health` | Liveness probe |
| POST | `/reset` | Start episode |
| POST | `/step` | Submit action |
| GET | `/state` | Current state |
| GET | `/tasks` | All tasks + schemas |
| GET | `/grader` | Last episode result |
| GET | `/baseline` | Run inference.py |
| POST | `/train` | Multi-episode loop |
| GET | `/analytics` | Learning curves |
| GET | `/curriculum` | Difficulty level |
| GET | `/generate` | Preview scenario |
| GET | `/docs` | API documentation |

---

## 📁 Project Structure

```
synapse/
├── README.md              ← This file (HF Spaces header included)
├── openenv.yaml           ← OpenEnv metadata
├── requirements.txt       ← Dependencies
├── pyproject.toml         ← uv run server entry point
├── Dockerfile             ← At ROOT (required — not in /server)
├── inference.py           ← Free HF token baseline script
├── demo.py                ← Interactive demo
├── models.py              ← Pydantic typed models
├── static/
│   └── index.html         ← Visual dashboard
└── server/
    ├── __init__.py
    ├── app.py             ← FastAPI + all endpoints
    ├── environment.py     ← Standard + investigation mode
    ├── generator.py       ← Procedural scenario generator
    ├── curriculum.py      ← Auto difficulty scaling
    ├── graders.py         ← Flexible keyword matching
    └── scenarios.py       ← 10 hardcoded base scenarios
```

---

## 📊 Baseline Scores (Free HF Inference)

| Task | Score |
|------|-------|
| task1 — Signal Monitor | 0.85 |
| task2 — Root Cause Engine | 0.70 |
| task3 — Priority Classifier | 0.80 |
| task4 — Remediation Planner | 0.55 |
| task5 — Post-Mortem Analyst | 0.45 |
| **Average** | **0.67** |

---

## 👥 Team Hive Mind

- **Marian Noah J** (Team Lead)
- **Diki Chumu Sherpa**

*OpenEnv Hackathon 2026 | Meta × PyTorch × Scaler School of Technology*