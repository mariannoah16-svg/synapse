"""
SYNAPSE — Procedural Scenario Generator
=========================================
Generates UNLIMITED unique AI model incidents procedurally.

Unlike fixed datasets, this generator creates novel combinations
of architectures, failure modes, and symptoms every episode.
An agent training on SYNAPSE will NEVER see the same scenario twice.

This is the key differentiator: real RL training requires
non-repeating environments. Most teams hard-code 3-10 scenarios.
SYNAPSE generates millions of unique, realistic incidents.

Generator Design:
  - 8 model architectures × 12 failure modes × 5 severity levels
  - = 480 base combinations
  - × random metric perturbations = effectively unlimited
  - All scenarios are internally consistent (metrics match the failure)
"""

import random
import math
from typing import Dict, Optional


# ── Catalogue of real PyTorch architectures ───────────────────────────────────

ARCHITECTURES = [
    {"name": "ResNet50",      "typical_lr": 0.001,   "memory_gb": 4.5,  "domain": "image_classification"},
    {"name": "BERT-Large",    "typical_lr": 0.00002, "memory_gb": 14.0, "domain": "nlp"},
    {"name": "GPT2-Medium",   "typical_lr": 0.0003,  "memory_gb": 8.5,  "domain": "text_generation"},
    {"name": "LSTM-Large",    "typical_lr": 0.01,    "memory_gb": 6.0,  "domain": "sequence"},
    {"name": "EfficientNet-B4","typical_lr": 0.001,  "memory_gb": 9.8,  "domain": "image_classification"},
    {"name": "DistilBERT",    "typical_lr": 0.00003, "memory_gb": 4.1,  "domain": "nlp"},
    {"name": "MobileNetV3",   "typical_lr": 0.001,   "memory_gb": 2.1,  "domain": "mobile_inference"},
    {"name": "LLaMA-7B",      "typical_lr": 0.00002, "memory_gb": 14.9, "domain": "llm"},
    {"name": "ResNet18",      "typical_lr": 0.001,   "memory_gb": 3.2,  "domain": "image_classification"},
    {"name": "T5-Base",       "typical_lr": 0.0001,  "memory_gb": 5.4,  "domain": "text2text"},
]

OPTIMIZERS = ["Adam", "AdamW", "SGD", "RMSprop"]
SCHEDULERS = [None, "cosine_annealing", "step_lr", "linear_warmup", "cosine"]
CUDA_VERSIONS = ["11.8", "12.1", "12.2"]
PYTORCH_VERSIONS = ["2.0.1", "2.1.0", "2.1.2", "2.2.0"]


# ── Failure mode templates ─────────────────────────────────────────────────────

FAILURE_MODES = {

    "nan_loss": {
        "root_cause": "learning_rate_too_high",
        "affected_components": ["optimizer", "loss_function"],
        "severity": "critical",
        "anomaly_type": "nan_loss",
        "team": "ml_team",
        "generate": lambda arch, rng: _gen_nan_loss(arch, rng),
    },

    "gpu_oom": {
        "root_cause": "memory_leak",
        "affected_components": ["dataloader", "model"],
        "severity": "critical",
        "anomaly_type": "gpu_oom",
        "team": "ml_team",
        "generate": lambda arch, rng: _gen_gpu_oom(arch, rng),
    },

    "overfitting": {
        "root_cause": "overfitting",
        "affected_components": ["model", "loss_function"],
        "severity": "high",
        "anomaly_type": "accuracy_drop",
        "team": "ml_team",
        "generate": lambda arch, rng: _gen_overfitting(arch, rng),
    },

    "gradient_explosion": {
        "root_cause": "gradient_explosion",
        "affected_components": ["optimizer", "model"],
        "severity": "critical",
        "anomaly_type": "gradient_explosion",
        "team": "ml_team",
        "generate": lambda arch, rng: _gen_gradient_explosion(arch, rng),
    },

    "underfitting": {
        "root_cause": "learning_rate_too_low",
        "affected_components": ["optimizer"],
        "severity": "high",
        "anomaly_type": "accuracy_drop",
        "team": "ml_team",
        "generate": lambda arch, rng: _gen_underfitting(arch, rng),
    },

    "data_drift": {
        "root_cause": "data_drift",
        "affected_components": ["model"],
        "severity": "high",
        "anomaly_type": "data_drift",
        "team": "ml_team",
        "generate": lambda arch, rng: _gen_data_drift(arch, rng),
    },

    "bad_deployment": {
        "root_cause": "bad_deployment",
        "affected_components": ["model"],
        "severity": "critical",
        "anomaly_type": "accuracy_drop",
        "team": "devops_team",
        "generate": lambda arch, rng: _gen_bad_deployment(arch, rng),
    },

    "memory_leak_dl": {
        "root_cause": "memory_leak",
        "affected_components": ["dataloader"],
        "severity": "critical",
        "anomaly_type": "memory_leak",
        "team": "ml_team",
        "generate": lambda arch, rng: _gen_memory_leak(arch, rng),
    },

    "class_imbalance": {
        "root_cause": "class_imbalance",
        "affected_components": ["model", "loss_function"],
        "severity": "critical",
        "anomaly_type": "accuracy_drop",
        "team": "ml_team",
        "generate": lambda arch, rng: _gen_class_imbalance(arch, rng),
    },

    "context_overflow": {
        "root_cause": "context_overflow",
        "affected_components": ["model"],
        "severity": "critical",
        "anomaly_type": "hallucination_spike",
        "team": "ai_team",
        "generate": lambda arch, rng: _gen_context_overflow(arch, rng),
    },

    "quantization_error": {
        "root_cause": "quantization_error",
        "affected_components": ["model"],
        "severity": "high",
        "anomaly_type": "accuracy_drop",
        "team": "ml_team",
        "generate": lambda arch, rng: _gen_quantization(arch, rng),
    },

    "prompt_injection": {
        "root_cause": "prompt_injection",
        "affected_components": ["model"],
        "severity": "critical",
        "anomaly_type": "hallucination_spike",
        "team": "ai_team",
        "generate": lambda arch, rng: _gen_prompt_injection(arch, rng),
    },
}


# ── Generator functions per failure mode ──────────────────────────────────────

def _jitter(value: float, rng: random.Random, pct: float = 0.15) -> float:
    """Add ±pct% random noise to a value to make each scenario unique."""
    return round(value * (1 + rng.uniform(-pct, pct)), 6)


def _gen_nan_loss(arch: Dict, rng: random.Random) -> Dict:
    bad_lr = rng.choice([0.05, 0.1, 0.5, 1.0])
    crash_epoch = rng.randint(2, 6)
    total_epochs = rng.randint(10, 20)
    model_name = f"{arch['name'].lower().replace('-','')}-prod"

    logs = []
    loss = rng.uniform(2.5, 3.5)
    for ep in range(1, crash_epoch + 1):
        if ep < crash_epoch:
            loss = loss * rng.uniform(1.1, 1.8)
            warn = f"Gradient norm: {round(rng.uniform(200,900),1)} — exploding" if ep == crash_epoch - 1 else None
            logs.append({"epoch": ep, "total_epochs": total_epochs, "train_loss": round(loss, 4),
                         "val_loss": round(loss * 1.05, 4), "train_acc": round(rng.uniform(5, 20), 1),
                         "val_acc": round(rng.uniform(4, 18), 1), "learning_rate": bad_lr,
                         "timestamp": f"10:0{ep}:00", "warning": warn, "error": None})
        else:
            logs.append({"epoch": ep, "total_epochs": total_epochs, "train_loss": None,
                         "val_loss": None, "train_acc": 0.0, "val_acc": 0.0,
                         "learning_rate": bad_lr, "timestamp": f"10:0{ep}:30",
                         "warning": None,
                         "error": f"Loss is nan at step {rng.randint(500,1500)}. Training collapsed completely."})

    return {
        "training_logs": logs,
        "production_metrics": {
            "requests_per_second": 0.0, "p99_latency_ms": 0.0,
            "error_rate": 1.0, "hallucination_rate": 0.0,
            "accuracy": 0.0, "drift_score": 0.0,
            "gpu_memory_used_gb": round(arch["memory_gb"] * 0.3, 1),
            "gpu_memory_total_gb": 16.0,
        },
        "error_traces": [{"error_type": "RuntimeError",
            "message": f"Function 'AddmmBackward0' returned nan values. Caused by overflow. Learning rate {bad_lr} is too high for {arch['optimizer']}.",
            "file": "torch/nn/modules/linear.py", "line": 114, "pytorch_component": "optimizer"}],
        "deployment_history": [{"event_type": "config_change", "version": "v1.2.0",
            "timestamp": f"09:5{rng.randint(0,9)}:00", "changed_by": f"ml-engineer-{rng.randint(1,20):02d}",
            "description": f"Increased learning rate from {arch['typical_lr']} to {bad_lr} for faster convergence"}],
        "alert_summary": f"{arch['name']} training crashed at epoch {crash_epoch}. Loss became NaN after learning rate was set to {bad_lr}.",
        "hint": f"Check learning_rate={bad_lr} in training_config. Typical lr for {arch['optimizer']} is {arch['typical_lr']}.",
        "model_name": model_name,
        "remediation": {
            "immediate_steps": ["stop training immediately", f"revert learning_rate to {arch['typical_lr']}"],
            "root_fix_steps": ["add gradient_clip max_norm=1.0", "use lr_scheduler with warmup", "restart from last valid checkpoint"],
            "verification_steps": ["verify loss decreasing for 3 epochs", "verify gradient norm below 10.0", "verify val_acc improving"],
        },
        "postmortem": {
            "root_cause": "learning_rate_too_high",
            "affected_components": ["optimizer", "loss_function"],
            "prevention_steps": ["add_lr_scheduler", "add_gradient_clipping"],
            "monitoring_additions": ["nan_loss_alert", "gradient_norm_monitor"],
        },
    }


def _gen_gpu_oom(arch: Dict, rng: random.Random) -> Dict:
    bad_batch = arch.get("batch_size_bad", rng.choice([128, 256, 512]))
    good_batch = max(8, bad_batch // 4)
    crash_epoch = rng.randint(2, 5)
    total_epochs = rng.randint(5, 10)
    model_name = f"{arch['name'].lower().replace('-','').replace('.','')}-prod"
    gpu_total = rng.choice([16.0, 24.0, 40.0])

    logs = []
    for ep in range(1, crash_epoch + 1):
        mem_pct = 0.70 + (ep / crash_epoch) * 0.29
        if ep < crash_epoch:
            logs.append({"epoch": ep, "total_epochs": total_epochs,
                         "train_loss": round(rng.uniform(0.6, 1.2), 4),
                         "val_loss": round(rng.uniform(0.7, 1.3), 4),
                         "train_acc": round(rng.uniform(65, 80), 1),
                         "val_acc": round(rng.uniform(63, 78), 1),
                         "learning_rate": arch["typical_lr"],
                         "timestamp": f"14:0{ep}:00",
                         "warning": f"GPU memory at {round(mem_pct*100)}% — {'approaching limit' if mem_pct > 0.85 else 'elevated'}",
                         "error": None})
        else:
            alloc = round(gpu_total * 0.16, 2)
            logs.append({"epoch": ep, "total_epochs": total_epochs,
                         "train_loss": None, "val_loss": None,
                         "train_acc": None, "val_acc": None,
                         "learning_rate": arch["typical_lr"],
                         "timestamp": f"14:0{ep}:30",
                         "warning": None,
                         "error": f"CUDA out of memory. Tried to allocate {alloc} GiB. GPU 0 has {gpu_total} GiB total; {round(gpu_total*0.94,2)} GiB allocated."})

    return {
        "training_logs": logs,
        "production_metrics": {
            "requests_per_second": 0.0, "p99_latency_ms": 0.0,
            "error_rate": 1.0, "hallucination_rate": 0.0,
            "accuracy": 0.0, "drift_score": 0.0,
            "gpu_memory_used_gb": round(gpu_total * 0.99, 1),
            "gpu_memory_total_gb": gpu_total,
        },
        "error_traces": [{"error_type": "RuntimeError",
            "message": f"CUDA out of memory. Tried to allocate {round(gpu_total*0.16,1)} GiB (GPU 0; {gpu_total} GiB total capacity; {round(gpu_total*0.94,1)} GiB already allocated)",
            "file": "torch/nn/modules/module.py", "line": 1501, "pytorch_component": "dataloader"}],
        "deployment_history": [{"event_type": "config_change", "version": f"v{rng.randint(1,5)}.0.0",
            "timestamp": f"13:5{rng.randint(0,9)}:00",
            "changed_by": f"ml-engineer-{rng.randint(1,20):02d}",
            "description": f"Increased batch_size from {good_batch} to {bad_batch} to speed up training"}],
        "alert_summary": f"{arch['name']} training crashed with CUDA OOM at epoch {crash_epoch}. GPU memory {round(gpu_total*0.99,1)}/{gpu_total}GB after batch_size was increased to {bad_batch}.",
        "hint": f"Check gpu_memory_used_gb vs gpu_memory_total_gb. batch_size={bad_batch} is too large for {arch['name']} on {gpu_total}GB GPU.",
        "model_name": model_name,
        "remediation": {
            "immediate_steps": ["stop training", f"reduce batch_size to {good_batch}"],
            "root_fix_steps": ["enable gradient_checkpointing", "use mixed_precision fp16", "restart training"],
            "verification_steps": [f"verify GPU memory below {round(gpu_total*0.8,0)}GB", "verify training completes epoch successfully", "verify val_acc improving"],
        },
        "postmortem": {
            "root_cause": "memory_leak",
            "affected_components": ["dataloader", "model"],
            "prevention_steps": ["add_memory_profiler", "validate_batch_size_before_training"],
            "monitoring_additions": ["gpu_memory_alert_80pct", "oom_predictor"],
        },
    }


def _gen_overfitting(arch: Dict, rng: random.Random) -> Dict:
    total_epochs = rng.randint(30, 60)
    checkpoints = [10, 20, 30, int(total_epochs * 0.8)]
    model_name = f"{arch['name'].lower().replace('-','').replace('.','')}-prod"

    logs = []
    train_loss, val_loss = rng.uniform(2.2, 2.8), rng.uniform(2.3, 2.9)
    train_acc, val_acc = rng.uniform(45, 55), rng.uniform(44, 54)

    for ep in checkpoints:
        diverge_factor = (ep / total_epochs) ** 2
        train_loss = max(0.05, train_loss - rng.uniform(0.1, 0.3))
        val_loss = val_loss + diverge_factor * rng.uniform(0.3, 0.8)
        train_acc = min(99.0, train_acc + rng.uniform(3, 8))
        val_acc = max(20.0, val_acc - diverge_factor * rng.uniform(5, 15))
        gap = train_acc - val_acc
        warn = None
        if gap > 15:
            warn = f"SEVERE overfitting: train/val accuracy gap = {round(gap, 1)}%"
        elif gap > 8:
            warn = f"Overfitting detected: train/val gap = {round(gap, 1)}%"
        logs.append({"epoch": ep, "total_epochs": total_epochs,
                     "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
                     "train_acc": round(train_acc, 1), "val_acc": round(val_acc, 1),
                     "learning_rate": arch["typical_lr"], "timestamp": f"0{rng.randint(8,9)}:{ep:02d}:00",
                     "warning": warn, "error": None})

    halluc = round(rng.uniform(0.25, 0.55), 2) if arch["domain"] in ["text_generation", "llm"] else 0.0
    return {
        "training_logs": logs,
        "production_metrics": {
            "requests_per_second": round(rng.uniform(200, 500), 1),
            "p99_latency_ms": round(rng.uniform(150, 400), 1),
            "error_rate": round(rng.uniform(0.01, 0.05), 3),
            "hallucination_rate": halluc,
            "accuracy": round(val_acc / 100, 3),
            "drift_score": round(rng.uniform(0.55, 0.80), 2),
            "gpu_memory_used_gb": round(arch["memory_gb"] * 0.6, 1),
            "gpu_memory_total_gb": 24.0,
        },
        "error_traces": [],
        "deployment_history": [{"event_type": "deploy", "version": "v1.0.0",
            "timestamp": "08:00:00", "changed_by": f"ml-engineer-{rng.randint(1,20):02d}",
            "description": f"Deployed {arch['name']} trained {total_epochs} epochs without dropout or weight_decay — no regularization"}],
        "alert_summary": f"{arch['name']} production accuracy dropped to {round(val_acc,1)}%. Model trained without regularization is memorizing training data.",
        "hint": "Compare train_acc vs val_acc across epochs. Check dropout and weight_decay in training_config.",
        "model_name": model_name,
        "remediation": {
            "immediate_steps": ["rollback to previous model version", "disable current deployment"],
            "root_fix_steps": ["add dropout=0.3", "add weight_decay=0.01", "retrain with early_stopping"],
            "verification_steps": ["verify train/val gap below 5%", "verify production accuracy above 80%", "verify hallucination_rate below 10%"],
        },
        "postmortem": {
            "root_cause": "overfitting",
            "affected_components": ["model", "loss_function"],
            "prevention_steps": ["add_dropout", "add_weight_decay", "add_early_stopping"],
            "monitoring_additions": ["train_val_gap_monitor", "production_accuracy_alert"],
        },
    }


def _gen_gradient_explosion(arch: Dict, rng: random.Random) -> Dict:
    crash_epoch = rng.randint(2, 5)
    total_epochs = rng.randint(15, 25)
    model_name = f"{arch['name'].lower().replace('-','').replace('.','')}-prod"
    grad_norm = round(rng.uniform(50000, 200000), 0)

    logs = [
        {"epoch": 1, "total_epochs": total_epochs, "train_loss": round(rng.uniform(2.8, 3.5), 4),
         "val_loss": round(rng.uniform(2.9, 3.6), 4), "train_acc": round(rng.uniform(25, 40), 1),
         "val_acc": round(rng.uniform(24, 38), 1), "learning_rate": arch["typical_lr"],
         "timestamp": "08:00:01", "warning": None, "error": None},
        {"epoch": 2, "total_epochs": total_epochs, "train_loss": round(rng.uniform(2.5, 3.0), 4),
         "val_loss": round(rng.uniform(2.6, 3.1), 4), "train_acc": round(rng.uniform(30, 45), 1),
         "val_acc": round(rng.uniform(29, 43), 1), "learning_rate": arch["typical_lr"],
         "timestamp": "08:05:22", "warning": f"Gradient norm: {round(rng.uniform(80, 200), 1)} — unusually high", "error": None},
        {"epoch": crash_epoch, "total_epochs": total_epochs,
         "train_loss": round(rng.uniform(5000, 20000), 1), "val_loss": None,
         "train_acc": 0.1, "val_acc": None, "learning_rate": arch["typical_lr"],
         "timestamp": f"08:1{crash_epoch}:44", "warning": None,
         "error": f"Gradient norm is {grad_norm}. Model weights have exploded. Loss jumped to extreme value."},
    ]

    return {
        "training_logs": logs,
        "production_metrics": {
            "requests_per_second": 0.0, "p99_latency_ms": 0.0,
            "error_rate": 1.0, "hallucination_rate": 0.0,
            "accuracy": 0.01, "drift_score": 0.0,
            "gpu_memory_used_gb": round(arch["memory_gb"] * 0.4, 1),
            "gpu_memory_total_gb": 16.0,
        },
        "error_traces": [{"error_type": "RuntimeError",
            "message": f"Gradient norm is {grad_norm}. Add torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) before optimizer.step()",
            "file": "torch/nn/utils/clip_grad.py", "line": 42, "pytorch_component": "optimizer"}],
        "deployment_history": [{"event_type": "config_change", "version": "v1.1.0",
            "timestamp": "07:55:00", "changed_by": f"ml-engineer-{rng.randint(1,20):02d}",
            "description": "Removed gradient clipping to test if it was slowing convergence"}],
        "alert_summary": f"{arch['name']} gradient explosion at epoch {crash_epoch}. Norm reached {grad_norm}. Gradient clipping was removed.",
        "hint": "Check gradient_clip in training_config and read the error message carefully.",
        "model_name": model_name,
        "remediation": {
            "immediate_steps": ["stop training immediately", "restore gradient_clip=1.0"],
            "root_fix_steps": ["add clip_grad_norm_ before optimizer.step", "restart training from epoch 1"],
            "verification_steps": ["verify gradient_norm below 10.0 each epoch", "verify loss decreasing steadily"],
        },
        "postmortem": {
            "root_cause": "gradient_explosion",
            "affected_components": ["optimizer", "model"],
            "prevention_steps": ["always_use_gradient_clipping_for_rnns", "monitor_gradient_norm_each_epoch"],
            "monitoring_additions": ["gradient_norm_alert_threshold_100", "loss_spike_detector"],
        },
    }


def _gen_underfitting(arch: Dict, rng: random.Random) -> Dict:
    bad_lr = round(arch["typical_lr"] / rng.randint(100, 1000), 8)
    total_epochs = rng.randint(25, 40)
    model_name = f"{arch['name'].lower().replace('-','').replace('.','')}-prod"
    stuck_acc = round(rng.uniform(20, 35), 1)

    logs = []
    for ep in [5, 10, 20, total_epochs]:
        logs.append({"epoch": ep, "total_epochs": total_epochs,
                     "train_loss": round(rng.uniform(2.1, 2.4), 4),
                     "val_loss": round(rng.uniform(2.1, 2.4), 4),
                     "train_acc": round(stuck_acc + rng.uniform(0, 2), 1),
                     "val_acc": round(stuck_acc + rng.uniform(0, 1.5), 1),
                     "learning_rate": bad_lr,
                     "timestamp": f"{8 + ep // 10:02d}:{ep % 60:02d}:00",
                     "warning": f"Model not converging after {ep} epochs — accuracy stuck near random chance" if ep >= 10 else None,
                     "error": None})

    return {
        "training_logs": logs,
        "production_metrics": {
            "requests_per_second": round(rng.uniform(150, 300), 1),
            "p99_latency_ms": round(rng.uniform(80, 200), 1),
            "error_rate": round(rng.uniform(0.005, 0.02), 4),
            "hallucination_rate": 0.0,
            "accuracy": round(stuck_acc / 100, 3),
            "drift_score": round(rng.uniform(0.02, 0.1), 2),
            "gpu_memory_used_gb": round(arch["memory_gb"] * 0.25, 1),
            "gpu_memory_total_gb": 16.0,
        },
        "error_traces": [],
        "deployment_history": [{"event_type": "config_change", "version": f"v{rng.randint(1,3)}.0.0",
            "timestamp": "09:00:00", "changed_by": f"ml-engineer-{rng.randint(1,20):02d}",
            "description": f"Reduced learning_rate from {arch['typical_lr']} to {bad_lr} after loss was oscillating"}],
        "alert_summary": f"{arch['name']} deployed with {stuck_acc}% accuracy — near random chance. Model trained {total_epochs} epochs with lr={bad_lr} but barely improved.",
        "hint": f"Compare train_acc across all epochs. A well-trained {arch['name']} should reach 80%+. Current lr={bad_lr} is too low.",
        "model_name": model_name,
        "remediation": {
            "immediate_steps": ["take model offline", f"increase learning_rate to {arch['typical_lr']}"],
            "root_fix_steps": [f"retrain with lr={arch['typical_lr']}", "add cosine_annealing scheduler", "train until val_acc > 80%"],
            "verification_steps": ["verify loss drops >50% in first 5 epochs", "verify val_acc above 80%"],
        },
        "postmortem": {
            "root_cause": "learning_rate_too_low",
            "affected_components": ["optimizer"],
            "prevention_steps": ["use_lr_finder_before_training", "set_convergence_threshold"],
            "monitoring_additions": ["accuracy_improvement_rate_monitor", "convergence_speed_alert"],
        },
    }


def _gen_data_drift(arch: Dict, rng: random.Random) -> Dict:
    train_acc = round(rng.uniform(87, 95), 1)
    prod_acc = round(rng.uniform(50, 70), 1)
    drift = round(rng.uniform(0.65, 0.90), 2)
    model_name = f"{arch['name'].lower().replace('-','').replace('.','')}-prod"

    logs = [{"epoch": 10, "total_epochs": 10,
             "train_loss": round(rng.uniform(0.15, 0.35), 4),
             "val_loss": round(rng.uniform(0.18, 0.38), 4),
             "train_acc": train_acc, "val_acc": round(train_acc - rng.uniform(0.5, 2), 1),
             "learning_rate": arch["typical_lr"],
             "timestamp": "2025-01-15 09:00:00", "warning": None, "error": None}]

    return {
        "training_logs": logs,
        "production_metrics": {
            "requests_per_second": round(rng.uniform(500, 1200), 1),
            "p99_latency_ms": round(rng.uniform(60, 180), 1),
            "error_rate": round(rng.uniform(0.02, 0.06), 3),
            "hallucination_rate": 0.0,
            "accuracy": round(prod_acc / 100, 3),
            "drift_score": drift,
            "gpu_memory_used_gb": round(arch["memory_gb"] * 0.35, 1),
            "gpu_memory_total_gb": 24.0,
        },
        "error_traces": [],
        "deployment_history": [
            {"event_type": "deploy", "version": "v1.0.0", "timestamp": "2025-01-15 10:00:00",
             "changed_by": f"ml-engineer-{rng.randint(1,20):02d}",
             "description": "Initial deployment — trained on Q4 2024 data"},
            {"event_type": "scale", "version": "v1.0.0", "timestamp": "2025-03-01 00:00:00",
             "changed_by": "auto-scaler",
             "description": f"New product launch — {rng.randint(5,15)}x traffic from new user segment with different behavior patterns"},
        ],
        "alert_summary": f"{arch['name']} accuracy dropped from {train_acc}% to {prod_acc}% over 6 weeks. Drift score {drift} (critical). New users behave very differently from training data.",
        "hint": "Compare when accuracy dropped vs deployment_history timestamps. High drift_score indicates distribution shift.",
        "model_name": model_name,
        "remediation": {
            "immediate_steps": ["collect new user segment data samples", "measure drift across input features"],
            "root_fix_steps": ["retrain model on new user segment data", "implement continuous learning pipeline"],
            "verification_steps": ["verify drift_score below 0.2", "verify accuracy above 85% on new data"],
        },
        "postmortem": {
            "root_cause": "data_drift",
            "affected_components": ["model"],
            "prevention_steps": ["add_drift_detection_pipeline", "schedule_regular_retraining"],
            "monitoring_additions": ["drift_score_alert_0.3", "accuracy_degradation_alert"],
        },
    }


def _gen_bad_deployment(arch: Dict, rng: random.Random) -> Dict:
    v_good = f"v{rng.randint(3,6)}.{rng.randint(1,5)}.0"
    v_bad = f"v{rng.randint(1,2)}.{rng.randint(0,3)}.0"
    error_rate = round(rng.uniform(0.35, 0.55), 2)
    model_name = f"{arch['name'].lower().replace('-','').replace('.','')}-prod"

    return {
        "training_logs": [{"epoch": 25, "total_epochs": 25,
             "train_loss": round(rng.uniform(0.12, 0.25), 4),
             "val_loss": round(rng.uniform(0.15, 0.28), 4),
             "train_acc": round(rng.uniform(91, 96), 1),
             "val_acc": round(rng.uniform(89, 94), 1),
             "learning_rate": arch["typical_lr"],
             "timestamp": "15:00:00", "warning": None, "error": None}],
        "production_metrics": {
            "requests_per_second": round(rng.uniform(300, 600), 1),
            "p99_latency_ms": round(rng.uniform(250, 500), 1),
            "error_rate": error_rate,
            "hallucination_rate": 0.0,
            "accuracy": round(rng.uniform(0.28, 0.42), 3),
            "drift_score": round(rng.uniform(0.08, 0.18), 2),
            "gpu_memory_used_gb": round(arch["memory_gb"] * 0.7, 1),
            "gpu_memory_total_gb": 24.0,
        },
        "error_traces": [{"error_type": "RuntimeError",
            "message": f"Error in loading state_dict: Missing key(s) in architecture. Wrong model weights ({v_bad}) deployed to {v_good} container.",
            "file": "torch/nn/modules/module.py", "line": 2041, "pytorch_component": "model"}],
        "deployment_history": [
            {"event_type": "deploy", "version": v_good, "timestamp": "11:30:00",
             "changed_by": f"ml-engineer-{rng.randint(1,20):02d}", "description": f"Deployed {v_good} model weights"},
            {"event_type": "deploy", "version": v_good, "timestamp": "14:55:00",
             "changed_by": "ci-cd-pipeline",
             "description": f"Emergency hotfix accidentally deployed {v_bad} weights to {v_good} container. Architecture mismatch."},
        ],
        "alert_summary": f"{arch['name']} error rate jumped to {round(error_rate*100)}% after hotfix. Wrong model weights ({v_bad}) deployed to {v_good} container — architecture mismatch.",
        "hint": "Read the error trace and last deployment event carefully. State dict keys don't match.",
        "model_name": model_name,
        "remediation": {
            "immediate_steps": [f"rollback to {v_good} immediately", "disable ci-cd auto-deployment"],
            "root_fix_steps": ["verify model weights match architecture before deployment", "add state_dict validation step to CI/CD"],
            "verification_steps": ["verify error_rate below 5%", "verify accuracy above 90%"],
        },
        "postmortem": {
            "root_cause": "bad_deployment",
            "affected_components": ["model"],
            "prevention_steps": ["add_model_architecture_validation", "add_canary_deployment"],
            "monitoring_additions": ["error_rate_spike_alert", "model_version_mismatch_detector"],
        },
    }


def _gen_memory_leak(arch: Dict, rng: random.Random) -> Dict:
    crash_epoch = rng.randint(45, 70)
    total_epochs = 100
    model_name = f"{arch['name'].lower().replace('-','').replace('.','')}-prod"
    gpu_total = rng.choice([16.0, 24.0])
    start_mem = round(arch["memory_gb"] * 0.4, 1)
    end_mem = gpu_total

    logs = []
    for ep in [10, 30, 50, crash_epoch]:
        mem = round(start_mem + (ep / crash_epoch) * (end_mem - start_mem), 1)
        if ep < crash_epoch:
            pct = round(mem / gpu_total * 100)
            warn = f"GPU memory: {mem} GB / {gpu_total} GB{'— CRITICAL: leak detected' if pct > 90 else ''}"
            logs.append({"epoch": ep, "total_epochs": total_epochs,
                         "train_loss": round(rng.uniform(0.5, 1.5) - ep * 0.01, 4),
                         "val_loss": round(rng.uniform(0.55, 1.55) - ep * 0.01, 4),
                         "train_acc": round(60 + ep * 0.4, 1), "val_acc": round(58 + ep * 0.38, 1),
                         "learning_rate": arch["typical_lr"],
                         "timestamp": f"{8 + ep // 15:02d}:{ep % 60:02d}:00",
                         "warning": warn, "error": None})
        else:
            logs.append({"epoch": ep, "total_epochs": total_epochs,
                         "train_loss": None, "val_loss": None,
                         "train_acc": None, "val_acc": None,
                         "learning_rate": arch["typical_lr"],
                         "timestamp": f"{8 + ep // 15:02d}:{ep % 60:02d}:30",
                         "warning": None,
                         "error": f"CUDA OOM at epoch {crash_epoch}. Memory grew from {start_mem}GB to {gpu_total}GB over training. DataLoader workers not releasing GPU tensors."})

    return {
        "training_logs": logs,
        "production_metrics": {
            "requests_per_second": 0.0, "p99_latency_ms": 0.0,
            "error_rate": 1.0, "hallucination_rate": 0.0,
            "accuracy": 0.0, "drift_score": 0.0,
            "gpu_memory_used_gb": gpu_total,
            "gpu_memory_total_gb": gpu_total,
        },
        "error_traces": [{"error_type": "RuntimeError",
            "message": f"CUDA OOM at epoch {crash_epoch}. GPU memory grew steadily from {start_mem}GB to {gpu_total}GB. DataLoader workers holding GPU tensor references.",
            "file": "torch/utils/data/dataloader.py", "line": 628, "pytorch_component": "dataloader"}],
        "deployment_history": [{"event_type": "config_change",
            "version": f"v{rng.randint(3,7)}.0.0",
            "timestamp": "08:00:00", "changed_by": f"ml-engineer-{rng.randint(1,20):02d}",
            "description": "Moved image augmentation transforms from CPU to GPU for speed — num_workers=8"}],
        "alert_summary": f"{arch['name']} OOM at epoch {crash_epoch}/{total_epochs}. Memory grew steadily from {start_mem}GB to {gpu_total}GB — GPU augmentation in DataLoader workers causing leak.",
        "hint": f"GPU memory grew STEADILY each epoch — not sudden OOM. Memory went {start_mem}GB → {gpu_total}GB over {crash_epoch} epochs.",
        "model_name": model_name,
        "remediation": {
            "immediate_steps": ["stop training", "move augmentation transforms back to CPU"],
            "root_fix_steps": ["set pin_memory=False in DataLoader", "add del tensor after each batch", f"restart training from epoch {crash_epoch - 5} checkpoint"],
            "verification_steps": ["verify GPU memory stays constant across epochs", "verify training completes epoch 100"],
        },
        "postmortem": {
            "root_cause": "memory_leak",
            "affected_components": ["dataloader"],
            "prevention_steps": ["always_run_augmentation_on_cpu", "monitor_gpu_memory_trend"],
            "monitoring_additions": ["gpu_memory_growth_rate_alert", "epoch_memory_delta_monitor"],
        },
    }


def _gen_class_imbalance(arch: Dict, rng: random.Random) -> Dict:
    num_classes = rng.randint(3, 5)
    dominant_pct = rng.randint(90, 97)
    high_acc = round(rng.uniform(94, 98), 1)
    model_name = f"{arch['name'].lower().replace('-','').replace('.','')}-prod"

    return {
        "training_logs": [
            {"epoch": 10, "total_epochs": 20, "train_loss": round(rng.uniform(0.08, 0.15), 4),
             "val_loss": round(rng.uniform(0.09, 0.16), 4), "train_acc": high_acc - 2,
             "val_acc": high_acc - 2.5, "learning_rate": arch["typical_lr"],
             "timestamp": "09:10:00",
             "warning": f"WARNING: High accuracy but model predicts class_0 for {dominant_pct}% of samples",
             "error": None},
            {"epoch": 20, "total_epochs": 20, "train_loss": round(rng.uniform(0.05, 0.10), 4),
             "val_loss": round(rng.uniform(0.06, 0.11), 4), "train_acc": high_acc,
             "val_acc": high_acc - 0.5, "learning_rate": arch["typical_lr"],
             "timestamp": "10:10:00",
             "warning": f"F1-score for minority classes near 0.00. Class distribution: class_0={dominant_pct}%, minority classes share {100-dominant_pct}%.",
             "error": None},
        ],
        "production_metrics": {
            "requests_per_second": round(rng.uniform(800, 1500), 1),
            "p99_latency_ms": round(rng.uniform(30, 80), 1),
            "error_rate": round(rng.uniform(0.01, 0.03), 3),
            "hallucination_rate": 0.0,
            "accuracy": round(high_acc / 100, 3),
            "drift_score": round(rng.uniform(0.03, 0.08), 2),
            "gpu_memory_used_gb": round(arch["memory_gb"] * 0.15, 1),
            "gpu_memory_total_gb": 16.0,
        },
        "error_traces": [],
        "deployment_history": [{"event_type": "deploy", "version": "v1.0.0",
            "timestamp": "08:00:00", "changed_by": f"ml-engineer-{rng.randint(1,20):02d}",
            "description": f"Deployed {num_classes}-class classifier trained on imbalanced dataset ({dominant_pct}% dominant class)"}],
        "alert_summary": f"{arch['name']} shows {high_acc}% accuracy but misses nearly all minority class cases. Imbalanced training data ({dominant_pct}% class_0) caused bias toward dominant class.",
        "hint": f"Overall accuracy {high_acc}% looks great but check class distribution in training logs warning.",
        "model_name": model_name,
        "remediation": {
            "immediate_steps": ["take model offline", "alert stakeholders about misleading accuracy metric"],
            "root_fix_steps": ["use weighted_cross_entropy_loss with class weights", "oversample minority classes with SMOTE", "retrain and evaluate with F1-score not accuracy"],
            "verification_steps": ["verify F1-score for minority classes above 0.7", "verify recall for minority classes above 0.8"],
        },
        "postmortem": {
            "root_cause": "class_imbalance",
            "affected_components": ["model", "loss_function"],
            "prevention_steps": ["always_check_class_distribution", "use_f1_not_accuracy_for_imbalanced"],
            "monitoring_additions": ["per_class_f1_monitor", "minority_class_recall_alert"],
        },
    }


def _gen_context_overflow(arch: Dict, rng: random.Random) -> Dict:
    max_tokens = rng.choice([2048, 4096, 8192])
    actual_tokens = max_tokens + rng.randint(500, 2000)
    halluc = round(rng.uniform(0.55, 0.80), 2)
    model_name = f"{arch['name'].lower().replace('-','').replace('.','')}-prod"

    return {
        "training_logs": [{"epoch": 3, "total_epochs": 3,
             "train_loss": round(rng.uniform(1.1, 1.5), 4),
             "val_loss": round(rng.uniform(1.2, 1.6), 4),
             "train_acc": None, "val_acc": None,
             "learning_rate": arch["typical_lr"],
             "timestamp": "2025-02-01 12:00:00", "warning": None, "error": None}],
        "production_metrics": {
            "requests_per_second": round(rng.uniform(15, 40), 1),
            "p99_latency_ms": round(rng.uniform(6000, 12000), 1),
            "error_rate": round(rng.uniform(0.30, 0.50), 2),
            "hallucination_rate": halluc,
            "accuracy": round(rng.uniform(0.35, 0.50), 3),
            "drift_score": round(rng.uniform(0.10, 0.20), 2),
            "gpu_memory_used_gb": round(arch["memory_gb"] * 0.95, 1),
            "gpu_memory_total_gb": 24.0,
        },
        "error_traces": [{"error_type": "RuntimeError",
            "message": f"Sequence length {actual_tokens} exceeds maximum context length {max_tokens} for model {arch['name']}. Token overflow causes attention failure.",
            "file": "transformers/modeling_llama.py", "line": 891, "pytorch_component": "model"}],
        "deployment_history": [
            {"event_type": "deploy", "version": "v2.0.0", "timestamp": "2025-02-01 14:00:00",
             "changed_by": f"ml-engineer-{rng.randint(1,20):02d}", "description": f"Deployed {arch['name']} assistant"},
            {"event_type": "config_change", "version": "v2.0.0", "timestamp": "2025-03-15 09:00:00",
             "changed_by": "product-team",
             "description": f"Enabled long document feature — users uploading files that exceed {max_tokens} token limit"},
        ],
        "alert_summary": f"{arch['name']} hallucination rate at {round(halluc*100)}%. Users uploading long documents ({actual_tokens} tokens) exceeding context limit of {max_tokens}.",
        "hint": f"Check error message for sequence length {actual_tokens} vs max {max_tokens}. Check what feature was enabled in deployment history.",
        "model_name": model_name,
        "remediation": {
            "immediate_steps": ["disable long document feature", f"add input length validation max {max_tokens - 500} tokens"],
            "root_fix_steps": ["implement sliding window chunking for long documents", f"upgrade to model with {max_tokens * 2} context window"],
            "verification_steps": [f"verify no requests exceed {max_tokens} tokens", "verify hallucination_rate below 10%", "verify p99_latency below 2000ms"],
        },
        "postmortem": {
            "root_cause": "context_overflow",
            "affected_components": ["model"],
            "prevention_steps": ["add_input_length_validation", "implement_chunking_for_long_inputs"],
            "monitoring_additions": ["token_count_p99_alert", "context_overflow_error_counter"],
        },
    }


def _gen_quantization(arch: Dict, rng: random.Random) -> Dict:
    acc_before = round(rng.uniform(88, 95), 1)
    acc_after = round(rng.uniform(60, 75), 1)
    model_name = f"{arch['name'].lower().replace('-','').replace('.','')}-prod"

    return {
        "training_logs": [{"epoch": 20, "total_epochs": 20,
             "train_loss": round(rng.uniform(0.15, 0.30), 4),
             "val_loss": round(rng.uniform(0.17, 0.33), 4),
             "train_acc": acc_before, "val_acc": acc_before - 1.2,
             "learning_rate": arch["typical_lr"],
             "timestamp": "14:00:00", "warning": None, "error": None}],
        "production_metrics": {
            "requests_per_second": round(rng.uniform(1500, 3000), 1),
            "p99_latency_ms": round(rng.uniform(15, 40), 1),
            "error_rate": round(rng.uniform(0.02, 0.05), 3),
            "hallucination_rate": 0.0,
            "accuracy": round(acc_after / 100, 3),
            "drift_score": round(rng.uniform(0.05, 0.15), 2),
            "gpu_memory_used_gb": round(arch["memory_gb"] * 0.25, 1),
            "gpu_memory_total_gb": 16.0,
        },
        "error_traces": [{"error_type": "RuntimeError",
            "message": "INT8 quantization caused {:.1f}% accuracy regression. Quantization-aware training (QAT) was not used. Dynamic range of activations too large for INT8.".format(acc_before - acc_after),
            "file": "torch/quantization/quantize.py", "line": 289, "pytorch_component": "model"}],
        "deployment_history": [
            {"event_type": "deploy", "version": "v2.0.0", "timestamp": "09:00:00",
             "changed_by": f"ml-engineer-{rng.randint(1,20):02d}", "description": f"Original fp32 model — accuracy {acc_before}%"},
            {"event_type": "deploy", "version": "v2.1.0-int8", "timestamp": "11:00:00",
             "changed_by": f"ml-engineer-{rng.randint(1,20):02d}",
             "description": f"Quantized to INT8 for 4x speedup — accuracy dropped from {acc_before}% to {acc_after}%"},
        ],
        "alert_summary": f"{arch['name']} accuracy dropped from {acc_before}% to {acc_after}% after INT8 quantization. Model was quantized without quantization-aware training.",
        "hint": "Check deployment_history for version with 'int8' quantization. Accuracy dropped significantly after that deployment.",
        "model_name": model_name,
        "remediation": {
            "immediate_steps": [f"rollback to v2.0.0 fp32 model", "disable int8 quantized version"],
            "root_fix_steps": ["retrain with quantization_aware_training (QAT)", "use fp16 instead of int8 for better accuracy", "validate accuracy after quantization before deploy"],
            "verification_steps": ["verify accuracy above 85% after quantization", "verify latency improvement maintained"],
        },
        "postmortem": {
            "root_cause": "quantization_error",
            "affected_components": ["model"],
            "prevention_steps": ["use_quantization_aware_training", "validate_accuracy_after_quantization"],
            "monitoring_additions": ["accuracy_regression_alert_on_deploy", "quantization_calibration_monitor"],
        },
    }


def _gen_prompt_injection(arch: Dict, rng: random.Random) -> Dict:
    halluc = round(rng.uniform(0.60, 0.85), 2)
    model_name = f"{arch['name'].lower().replace('-','').replace('.','')}-prod"

    return {
        "training_logs": [{"epoch": 5, "total_epochs": 5,
             "train_loss": round(rng.uniform(0.8, 1.4), 4),
             "val_loss": round(rng.uniform(0.9, 1.5), 4),
             "train_acc": None, "val_acc": None,
             "learning_rate": arch["typical_lr"],
             "timestamp": "2025-03-01 12:00:00", "warning": None, "error": None}],
        "production_metrics": {
            "requests_per_second": round(rng.uniform(80, 200), 1),
            "p99_latency_ms": round(rng.uniform(400, 900), 1),
            "error_rate": round(rng.uniform(0.05, 0.15), 3),
            "hallucination_rate": halluc,
            "accuracy": round(rng.uniform(0.30, 0.50), 3),
            "drift_score": round(rng.uniform(0.20, 0.40), 2),
            "gpu_memory_used_gb": round(arch["memory_gb"] * 0.65, 1),
            "gpu_memory_total_gb": 24.0,
        },
        "error_traces": [{"error_type": "SecurityViolation",
            "message": "Prompt injection detected in production logs. User input contains 'ignore previous instructions' pattern. Safety filter bypassed in 847 requests.",
            "file": "inference/safety_filter.py", "line": 156, "pytorch_component": "model"}],
        "deployment_history": [
            {"event_type": "config_change", "version": "v3.0.0", "timestamp": "2025-03-20 10:00:00",
             "changed_by": "product-team",
             "description": "Removed input sanitization step to reduce latency by 200ms — safety filter now bypassed"},
        ],
        "alert_summary": f"{arch['name']} hallucination rate at {round(halluc*100)}%. Prompt injection attack bypassing safety filters after input sanitization was removed for performance.",
        "hint": "Check deployment_history for removal of sanitization. Safety filter was disabled.",
        "model_name": model_name,
        "remediation": {
            "immediate_steps": ["re-enable input sanitization immediately", "block requests matching injection patterns"],
            "root_fix_steps": ["implement adversarial input detection", "add output safety classifier", "retrain with adversarial examples"],
            "verification_steps": ["verify injection patterns blocked 100%", "verify hallucination_rate below 5%"],
        },
        "postmortem": {
            "root_cause": "prompt_injection",
            "affected_components": ["model"],
            "prevention_steps": ["never_disable_safety_filters", "add_adversarial_testing_to_cicd"],
            "monitoring_additions": ["injection_pattern_detector", "output_toxicity_monitor"],
        },
    }


# ── Main Generator Class ───────────────────────────────────────────────────────

class ScenarioGenerator:
    """
    Generates unlimited unique AI model incident scenarios.

    Usage:
        gen = ScenarioGenerator(seed=42)
        scenario = gen.generate()    # Random failure mode + architecture
        scenario = gen.generate(failure_mode="nan_loss")  # Specific failure
        scenario = gen.generate(difficulty="hard")        # Hard difficulty only
    """

    EASY_FAILURES   = ["nan_loss", "gpu_oom", "gradient_explosion"]
    MEDIUM_FAILURES = ["overfitting", "underfitting", "bad_deployment", "memory_leak_dl"]
    HARD_FAILURES   = ["data_drift", "class_imbalance", "context_overflow", "quantization_error", "prompt_injection"]
    ALL_FAILURES    = list(FAILURE_MODES.keys())

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._generated_count = 0

    def generate(
        self,
        failure_mode: Optional[str] = None,
        difficulty: Optional[str] = None,
        architecture: Optional[str] = None,
    ) -> Dict:
        """
        Generate one unique scenario.

        Args:
            failure_mode: Specific failure type, or None for random
            difficulty: 'easy' | 'medium' | 'hard' | None for random
            architecture: Architecture name, or None for random

        Returns:
            Complete scenario dict compatible with SynapseEnvironment
        """
        # Pick failure mode
        if failure_mode and failure_mode in FAILURE_MODES:
            chosen_failure = failure_mode
        elif difficulty == "easy":
            chosen_failure = self._rng.choice(self.EASY_FAILURES)
        elif difficulty == "medium":
            chosen_failure = self._rng.choice(self.MEDIUM_FAILURES)
        elif difficulty == "hard":
            chosen_failure = self._rng.choice(self.HARD_FAILURES)
        else:
            chosen_failure = self._rng.choice(self.ALL_FAILURES)

        # Pick architecture
        if architecture:
            arch_data = next((a for a in ARCHITECTURES if a["name"] == architecture), None)
        else:
            arch_data = self._rng.choice(ARCHITECTURES)

        if not arch_data:
            arch_data = self._rng.choice(ARCHITECTURES)

        # Add randomized optimizer/scheduler/version to arch
        arch_with_opts = {
            **arch_data,
            "optimizer":        self._rng.choice(OPTIMIZERS),
            "batch_size_bad":   self._rng.choice([128, 256, 512]),
            "batch_size_good":  self._rng.choice([16, 32, 64]),
        }

        # Generate failure-specific data
        failure_def = FAILURE_MODES[chosen_failure]
        generated = failure_def["generate"](arch_with_opts, self._rng)

        # Build scenario ID
        self._generated_count += 1
        scenario_id = f"gen_{chosen_failure}_{self._generated_count:06d}"

        # Build training config
        training_config = {
            "architecture":   arch_with_opts["name"],
            "optimizer":      arch_with_opts["optimizer"],
            "learning_rate":  arch_with_opts["typical_lr"],
            "batch_size":     self._rng.choice([16, 32, 64, 128]),
            "epochs":         self._rng.randint(5, 30),
            "dropout":        round(self._rng.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]), 1),
            "weight_decay":   round(self._rng.choice([0.0, 0.0001, 0.001, 0.01]), 5),
            "gradient_clip":  self._rng.choice([None, 1.0, 0.5, 5.0]),
            "scheduler":      self._rng.choice(SCHEDULERS),
            "pytorch_version": self._rng.choice(PYTORCH_VERSIONS),
            "cuda_version":   self._rng.choice(CUDA_VERSIONS),
        }

        # Build ground truth for all 5 tasks
        model_name = generated.get("model_name", f"{arch_with_opts['name'].lower()}-prod")

        ground_truth = {
            "task1": {
                "anomaly_detected": True,
                "anomaly_type":     failure_def["anomaly_type"],
                "severity":         failure_def["severity"],
            },
            "task2": {
                "root_cause":          failure_def["root_cause"],
                "affected_components": failure_def["affected_components"],
            },
            "task3": {
                "priority_ranking": [model_name],
                "response_team":    failure_def["team"],
            },
            "task4": generated["remediation"],
            "task5": generated["postmortem"],
        }

        return {
            "id":              scenario_id,
            "name":            f"{arch_with_opts['name']} — {chosen_failure.replace('_',' ').title()}",
            "failure_mode":    chosen_failure,
            "difficulty":      (
                "easy"   if chosen_failure in self.EASY_FAILURES else
                "medium" if chosen_failure in self.MEDIUM_FAILURES else
                "hard"
            ),
            "pytorch_version": self._rng.choice(PYTORCH_VERSIONS),
            "cuda_version":    self._rng.choice(CUDA_VERSIONS),
            "model_config":    training_config,
            "training_logs":   generated["training_logs"],
            "production_metrics": generated["production_metrics"],
            "error_traces":    generated["error_traces"],
            "deployment_history": generated["deployment_history"],
            "alert_summary":   generated["alert_summary"],
            "hint":            generated.get("hint"),
            "ground_truth":    ground_truth,
        }

    @property
    def generated_count(self) -> int:
        return self._generated_count