        """
SYNAPSE v4.0 - Multi-Turn Investigation Environment
"""
from __future__ import annotations
import random
import json
from typing import Dict, List, Optional, Tuple

from models import (
    Action, Observation, Reward,
    ModelConfig, ProductionMetrics, ErrorTrace,
    DeploymentEvent, PyTorchTrainingLog,
)
from server.generator import ScenarioGenerator
from server.graders import GRADERS, grade_task1, grade_task2, grade_task4
from server.curriculum import CurriculumManager
from server.scenarios import SCENARIOS as HARDCODED_SCENARIOS

VALID_TASKS  = ["task1", "task2", "task3", "task4", "task5"]
MAX_STEPS    = 10
STEP_PENALTY = 0.01

class SynapseEnvironment:
    def __init__(self, seed=None, use_curriculum=True):
        self._rng        = random.Random(seed)
        self._seed       = seed
        self._generator  = ScenarioGenerator(seed=seed)
        self._curriculum = CurriculumManager() if use_curriculum else None
        self._scenario   = None
        self._task_id    = None
        self._step       = 0
        self._done       = False
        self._last_reward = None
        self._episode_id  = 0
        self._investigation_mode  = False
        self._investigation_phase = 0
        self._investigation_scores = []
        self._cumulative_score    = 0.0
        self._total_episodes = 0
        self._total_score    = 0.0

    def reset(self, task_id=None, difficulty=None, use_generated=True, mode="standard"):
        if self._curriculum and not difficulty:
            difficulty = self._curriculum.current_difficulty
        if use_generated:
            self._scenario = self._generator.generate(difficulty=difficulty)
        else:
            self._scenario = self._rng.choice(HARDCODED_SCENARIOS)
        self._step       = 0
        self._done       = False
        self._last_reward = None
        self._episode_id += 1
        self._investigation_mode   = (mode == "investigation")
        self._investigation_phase  = 0
        self._investigation_scores = []
        self._cumulative_score     = 0.0
        if self._investigation_mode:
            self._task_id = "investigation"
            return self._build_obs(phase=0)
        else:
            self._task_id = task_id if task_id in VALID_TASKS else self._rng.choice(VALID_TASKS)
            return self._build_obs()

    def step(self, action):
        if self._scenario is None:
            raise RuntimeError("Call reset() first.")
        if self._done:
            raise RuntimeError("Episode done. Call reset().")
        self._step += 1
        if self._investigation_mode:
            return self._step_investigation(action)
        return self._step_standard(action)

    def state(self):
        curriculum_stats = self._curriculum.get_stats() if self._curriculum else None
        base = {
            "episode_id": self._episode_id,
            "task_id": self._task_id,
            "mode": "investigation" if self._investigation_mode else "standard",
            "step": self._step,
            "max_steps": MAX_STEPS,
            "done": self._done,
            "scenario_id": self._scenario["id"] if self._scenario else None,
            "scenario_name": self._scenario["name"] if self._scenario else None,
            "scenario_difficulty": self._scenario.get("difficulty") if self._scenario else None,
            "failure_mode": self._scenario.get("failure_mode") if self._scenario else None,
            "last_reward": self._last_reward.model_dump() if self._last_reward else None,
            "total_episodes": self._total_episodes,
            "overall_avg_score": round(self._total_score / self._total_episodes, 4) if self._total_episodes else 0.0,
            "curriculum": curriculum_stats,
            "scenarios_generated": self._generator.generated_count,
        }
        if self._investigation_mode:
            phase_names = ["triage", "diagnosis", "resolution"]
            base["investigation"] = {
                "current_phase": self._investigation_phase,
                "phase_name": phase_names[min(self._investigation_phase, 2)],
                "phase_scores": self._investigation_scores,
                "cumulative_score": round(self._cumulative_score, 4),
            }
        return base

    def get_tasks(self):
        return [
            {"task_id": "task1", "name": "Signal Monitor", "difficulty": "easy",
             "expected_baseline_score": 0.85,
             "description": "Detect anomaly type and severity from PyTorch logs.",
             "action_schema": {"anomaly_detected": "bool",
                "anomaly_type": "nan_loss|loss_spike|accuracy_drop|gpu_oom|data_drift|latency_spike|hallucination_spike|gradient_explosion|memory_leak|none",
                "severity": "low|medium|high|critical"}},
            {"task_id": "task2", "name": "Root Cause Engine", "difficulty": "medium",
             "expected_baseline_score": 0.70,
             "description": "Diagnose root cause and affected PyTorch components.",
             "action_schema": {"root_cause": "learning_rate_too_high|learning_rate_too_low|overfitting|underfitting|data_drift|quantization_error|prompt_injection|context_overflow|bad_deployment|class_imbalance|gradient_explosion|memory_leak",
                "affected_components": "list[str]"}},
            {"task_id": "task3", "name": "Priority Classifier", "difficulty": "medium-hard",
             "expected_baseline_score": 0.65,
             "description": "Classify urgency and assign response team.",
             "action_schema": {"priority_ranking": "list[str]", "response_team": "ml_team|ai_team|devops_team|all_hands"}},
            {"task_id": "task4", "name": "Remediation Planner", "difficulty": "hard",
             "expected_baseline_score": 0.55,
             "description": "3-phase remediation: immediate, root fix, verification.",
             "action_schema": {"immediate_steps": "list[str]", "root_fix_steps": "list[str]", "verification_steps": "list[str]"}},
            {"task_id": "task5", "name": "Post-Mortem Analyst", "difficulty": "very-hard",
             "expected_baseline_score": 0.45,
             "description": "Structured post-mortem report.",
             "action_schema": {"postmortem": {"root_cause": "str", "affected_components": "list[str]", "prevention_steps": "list[str]", "monitoring_additions": "list[str]"}}},
            {"task_id": "investigation", "name": "Full Investigation (Multi-Turn)", "difficulty": "progressive",
             "expected_baseline_score": 0.65,
             "description": "3-phase episode: Triage(0.30) → Diagnosis(0.30) → Resolution(0.40). Partial reward each step.",
             "action_schema": {
                "phase_1_triage":    {"anomaly_detected": "bool", "anomaly_type": "str", "severity": "str"},
                "phase_2_diagnosis": {"root_cause": "str", "affected_components": "list[str]"},
                "phase_3_resolution": {"immediate_steps": "list[str]", "root_fix_steps": "list[str]", "verification_steps": "list[str]"},
             }},
        ]

    def get_analytics(self):
        curriculum_stats = self._curriculum.get_stats() if self._curriculum else {}
        from server.generator import FAILURE_MODES, ARCHITECTURES
        return {
            "environment": "SYNAPSE v4.0",
            "total_episodes": self._total_episodes,
            "overall_avg_score": round(self._total_score / self._total_episodes, 4) if self._total_episodes else 0.0,
            "scenarios_generated": self._generator.generated_count,
            "curriculum": curriculum_stats,
            "failure_modes": list(FAILURE_MODES.keys()),
            "architectures": [a["name"] for a in ARCHITECTURES],
            "modes": ["standard", "investigation"],
        }

    def _step_standard(self, action):
        gt     = self._scenario["ground_truth"][self._task_id]
        grader = GRADERS[self._task_id]
        reward = grader(action, gt)
        if self._step > 1:
            penalty = round(STEP_PENALTY * (self._step - 1), 4)
            reward.score = round(max(0.001, reward.score - penalty), 4)
            reward.breakdown["step_penalty"] = -penalty
            reward.feedback += f" | step_penalty -{penalty}"
        self._done = True
        self._last_reward = reward
        self._total_episodes += 1
        self._total_score    += reward.score
        if self._curriculum:
            self._curriculum.record(reward.score, self._task_id, self._scenario.get("difficulty", "easy"))
        info = {
            "episode_id": self._episode_id, "mode": "standard",
            "task_id": self._task_id, "scenario_id": self._scenario["id"],
            "scenario_difficulty": self._scenario.get("difficulty"),
            "failure_mode": self._scenario.get("failure_mode"),
            "steps_used": self._step,
        }
        return self._build_obs(), reward, self._done, info

    def _step_investigation(self, action):
        phase = self._investigation_phase
        gt    = self._scenario["ground_truth"]
        PHASES = [
            {"name": "triage",    "weight": 0.30, "grader": grade_task1, "gt_key": "task1",
             "next_hint": "Phase 1 done. Training logs + error traces now visible."},
            {"name": "diagnosis", "weight": 0.30, "grader": grade_task2, "gt_key": "task2",
             "next_hint": "Phase 2 done. Full deployment history now visible."},
            {"name": "resolution","weight": 0.40, "grader": grade_task4, "gt_key": "task4",
             "next_hint": "Investigation complete."},
        ]
        p           = PHASES[phase]
        phase_r     = p["grader"](action, gt[p["gt_key"]])
        weighted    = round(phase_r.score * p["weight"], 4)
        self._investigation_scores.append(round(phase_r.score, 4))
        self._cumulative_score = round(self._cumulative_score + weighted, 4)

        reward = Reward(
            score=self._cumulative_score,
            breakdown={
                f"phase_{phase+1}_{p['name']}": weighted,
                "cumulative": self._cumulative_score,
                "phases_done": phase + 1,
                "phases_total": 3,
            },
            feedback=(
                f"Phase {phase+1}/3 ({p['name']}): "
                f"score={phase_r.score:.3f} × weight={p['weight']} = {weighted:.3f} | "
                f"cumulative={self._cumulative_score:.3f} | "
                f"{p['next_hint']}"
            ),
        )

        self._investigation_phase += 1
        done = self._investigation_phase >= 3
        self._done = done

        if done:
            self._last_reward    = reward
            self._total_episodes += 1
            self._total_score    += self._cumulative_score
            if self._curriculum:
                self._curriculum.record(self._cumulative_score, "investigation", self._scenario.get("difficulty", "easy"))

        next_phase = min(self._investigation_phase, 2)
        obs = self._build_obs(phase=next_phase if not done else 2)

        info = {
            "episode_id": self._episode_id, "mode": "investigation",
            "phase_completed": phase + 1, "phase_name": p["name"],
            "phase_score": round(phase_r.score, 4),
            "weighted_score": weighted, "cumulative_score": self._cumulative_score,
            "phases_remaining": max(0, 2 - phase),
            "scenario_id": self._scenario["id"],
            "scenario_difficulty": self._scenario.get("difficulty"),
            "failure_mode": self._scenario.get("failure_mode"),
        }
        return obs, reward, done, info

    def _build_obs(self, phase=None):
        s   = self._scenario
        cfg = s["model_config"]

        # Calculate Phase Context
        current_phase_idx = self._investigation_phase if phase is None else phase
        phase_context = {
            "phase_number": current_phase_idx + 1,
            "phase_name": ["triage", "diagnosis", "resolution"][min(current_phase_idx, 2)],
            "data_available": {
                "production_metrics": True,
                "training_logs": current_phase_idx >= 1,
                "error_traces": current_phase_idx >= 1,
                "deployment_history": current_phase_idx >= 2,
            },
            "cumulative_score": round(self._cumulative_score, 4)
        }

        if self._investigation_mode and phase is not None:
            if phase == 0:
                logs = []; errors = []; deploys = []
                base_hint = (
                    "PHASE 1 — TRIAGE: You see only production metrics and the alert. "
                    "Submit: anomaly_detected, anomaly_type, severity. "
                    "Training logs unlock after this step."
                )
            elif phase == 1:
                logs    = [PyTorchTrainingLog(**l) for l in s["training_logs"]]
                errors  = [ErrorTrace(**e) for e in s["error_traces"]]
                deploys = []
                base_hint = (
                    "PHASE 2 — DIAGNOSIS: Training logs + error traces unlocked. "
                    "Submit: root_cause, affected_components. "
                    "Deployment history unlocks after this step."
                )
            else:
                logs    = [PyTorchTrainingLog(**l) for l in s["training_logs"]]
                errors  = [ErrorTrace(**e) for e in s["error_traces"]]
                deploys = [DeploymentEvent(**d) for d in s["deployment_history"]]
                base_hint = (
                    "PHASE 3 — RESOLUTION: Full context available. "
                    "Submit: immediate_steps, root_fix_steps, verification_steps."
                )

            # Embed phase_context directly into the hint string as formatted JSON
            hint = f"{base_hint}\n\n[PHASE CONTEXT]\n{json.dumps(phase_context, indent=2)}"

        else:
            logs    = [PyTorchTrainingLog(**l) for l in s["training_logs"]]
            errors  = [ErrorTrace(**e) for e in s["error_traces"]]
            deploys = [DeploymentEvent(**d) for d in s["deployment_history"]]
            base_hint = s.get("hint") if self._task_id == "task1" else None

            # Append context to standard tasks as well if a hint exists
            if base_hint:
                hint = f"{base_hint}\n\n[PHASE CONTEXT]\n{json.dumps(phase_context, indent=2)}"
            else:
                hint = f"[PHASE CONTEXT]\n{json.dumps(phase_context, indent=2)}"

        return Observation(
            task_id=self._task_id, step=self._step, max_steps=MAX_STEPS,
            pytorch_version=s["pytorch_version"], cuda_version=s.get("cuda_version"),
            training_config=ModelConfig(**cfg),
            training_logs=logs,
            production_metrics=ProductionMetrics(**s["production_metrics"]),
            error_traces=errors,
            deployment_history=deploys,
            alert_summary=s["alert_summary"],
            hint=hint,
        )
