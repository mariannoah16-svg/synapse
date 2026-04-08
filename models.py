"""
SYNAPSE — Typed Data Models
============================
OpenEnv-compliant Pydantic models defining all data structures.

Environment: SYNAPSE (Systematic Neural Analysis and Production Supervision Environment)
Domain: AI Model Incident Response in Production Systems

These models define the contract between the environment and any AI agent.
"""

from __future__ import annotations
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models: Components of the Observation
# ─────────────────────────────────────────────────────────────────────────────

class PyTorchTrainingLog(BaseModel):
    """A single epoch entry from a PyTorch training run."""
    epoch: int = Field(description="Current epoch number")
    total_epochs: int = Field(description="Total planned epochs")
    train_loss: Optional[float] = Field(None, description="Training loss (None if crashed)")
    val_loss: Optional[float] = Field(None, description="Validation loss")
    train_acc: Optional[float] = Field(None, description="Training accuracy 0-100%")
    val_acc: Optional[float] = Field(None, description="Validation accuracy 0-100%")
    learning_rate: float = Field(description="Current learning rate")
    timestamp: str = Field(description="HH:MM:SS when this epoch completed")
    warning: Optional[str] = Field(None, description="PyTorch warning message if any")
    error: Optional[str] = Field(None, description="PyTorch error message if any")


class ModelConfig(BaseModel):
    """PyTorch model training configuration."""
    architecture: str = Field(description="e.g. ResNet50, BERT-Large, GPT2")
    optimizer: str = Field(description="e.g. Adam, AdamW, SGD")
    learning_rate: float = Field(description="Optimizer learning rate")
    batch_size: int = Field(description="Training batch size")
    epochs: int = Field(description="Total training epochs planned")
    dropout: float = Field(description="Dropout rate 0.0-1.0")
    weight_decay: float = Field(description="L2 regularization weight decay")
    gradient_clip: Optional[float] = Field(None, description="Max gradient norm for clipping")
    scheduler: Optional[str] = Field(None, description="LR scheduler type if used")
    pytorch_version: str = Field(description="PyTorch version e.g. 2.1.0")
    cuda_version: Optional[str] = Field(None, description="CUDA version if GPU training")


class ProductionMetrics(BaseModel):
    """Real-time production metrics for a deployed AI model."""
    requests_per_second: float = Field(description="Current RPS throughput")
    p99_latency_ms: float = Field(description="99th percentile latency in milliseconds")
    error_rate: float = Field(description="Fraction of requests returning errors 0.0-1.0")
    hallucination_rate: float = Field(description="Fraction of responses that are hallucinated 0.0-1.0")
    accuracy: float = Field(description="Current model accuracy 0.0-1.0")
    drift_score: float = Field(description="Data drift severity 0.0-1.0 (higher=worse)")
    gpu_memory_used_gb: float = Field(description="GPU memory currently allocated in GB")
    gpu_memory_total_gb: float = Field(description="Total GPU memory available in GB")


class ErrorTrace(BaseModel):
    """A Python/PyTorch stack trace entry from the inference server."""
    error_type: str = Field(description="Exception class e.g. RuntimeError, ValueError")
    message: str = Field(description="Full error message text")
    file: str = Field(description="Python file where error occurred")
    line: int = Field(description="Line number of the error")
    pytorch_component: str = Field(description="PyTorch component e.g. optimizer, dataloader, model")


class DeploymentEvent(BaseModel):
    """A record of a deployment or configuration change."""
    event_type: str = Field(description="deploy | rollback | config_change | scale")
    version: str = Field(description="Model or service version e.g. v2.1.0")
    timestamp: str = Field(description="When the event occurred HH:MM:SS")
    changed_by: str = Field(description="Engineer or system that made the change")
    description: str = Field(description="Human-readable description of what changed")


# ─────────────────────────────────────────────────────────────────────────────
# Core OpenEnv Models: Observation, Action, Reward
# ─────────────────────────────────────────────────────────────────────────────

class Observation(BaseModel):
    """
    What the AI agent sees at each step.

    Contains complete incident context from a production AI system:
    - PyTorch training logs showing model behavior over time
    - Real-time production metrics (accuracy, latency, drift)
    - Error stack traces from the inference server
    - Deployment history showing recent changes
    """
    # Task context
    task_id: str = Field(description="Which task to solve: task1 through task5")
    step: int = Field(description="Current step number within this episode")
    max_steps: int = Field(description="Maximum steps allowed before episode ends")

    # PyTorch training context
    pytorch_version: str = Field(description="PyTorch version used for training")
    cuda_version: Optional[str] = Field(None, description="CUDA version if applicable")
    training_config: ModelConfig = Field(description="Model architecture and training hyperparameters")
    training_logs: List[PyTorchTrainingLog] = Field(description="Epoch-by-epoch training history")

    # Production monitoring data
    production_metrics: ProductionMetrics = Field(description="Current production system health")
    error_traces: List[ErrorTrace] = Field(description="Recent error stack traces from inference server")
    deployment_history: List[DeploymentEvent] = Field(description="Recent deployments and config changes")

    # Incident summary
    alert_summary: str = Field(description="Plain English description of the current incident")
    hint: Optional[str] = Field(None, description="Helpful hint shown only in Task 1 (easy mode)")


class Action(BaseModel):
    """
    What the AI agent does in response to an observation.

    Different tasks use different fields — only fill the fields
    relevant to the current task_id.

    Task 1 (Signal Monitor):    anomaly_detected, anomaly_type, severity
    Task 2 (Root Cause Engine): root_cause, affected_components
    Task 3 (Priority Classifier): priority_ranking, response_team
    Task 4 (Remediation Planner): immediate_steps, root_fix_steps, verification_steps
    Task 5 (Post-Mortem Analyst): postmortem dict
    """
    # Task 1 — Signal Monitor (Easy)
    anomaly_detected: Optional[bool] = Field(None, description="True if any anomaly detected in the system")
    anomaly_type: Optional[Literal[
        "nan_loss", "loss_spike", "accuracy_drop", "gpu_oom",
        "data_drift", "latency_spike", "hallucination_spike",
        "gradient_explosion", "memory_leak", "none"
    ]] = Field(None, description="Type of anomaly detected in PyTorch logs or production metrics")
    severity: Optional[Literal["low", "medium", "high", "critical"]] = Field(
        None, description="Severity level of the incident"
    )

    # Task 2 — Root Cause Engine (Medium)
    root_cause: Optional[Literal[
        "learning_rate_too_high", "learning_rate_too_low",
        "overfitting", "underfitting", "data_drift",
        "quantization_error", "prompt_injection",
        "context_overflow", "bad_deployment",
        "class_imbalance", "gradient_explosion", "memory_leak"
    ]] = Field(None, description="Root cause of the incident")
    affected_components: Optional[List[str]] = Field(
        None, description="PyTorch components affected e.g. ['optimizer', 'dataloader']"
    )

    # Task 3 — Priority Classifier (Medium-Hard)
    priority_ranking: Optional[List[str]] = Field(
        None, description="Model names ordered from highest to lowest priority"
    )
    response_team: Optional[Literal[
        "ml_team", "ai_team", "devops_team", "all_hands"
    ]] = Field(None, description="Team responsible for resolving this incident")

    # Task 4 — Remediation Planner (Hard)
    immediate_steps: Optional[List[str]] = Field(
        None, description="Immediate actions to stop the incident from worsening"
    )
    root_fix_steps: Optional[List[str]] = Field(
        None, description="Steps to permanently fix the root cause"
    )
    verification_steps: Optional[List[str]] = Field(
        None, description="Steps to confirm the system has recovered"
    )

    # Task 5 — Post-Mortem Analyst (Very Hard)
    postmortem: Optional[Dict] = Field(
        None,
        description="""Structured post-mortem report. Must contain:
        {
          'root_cause': str,
          'affected_components': list[str],
          'prevention_steps': list[str],
          'monitoring_additions': list[str]
        }"""
    )


class Reward(BaseModel):
    """
    Grader output for one agent action.

    Score is always between 0.0 (completely wrong) and 1.0 (perfect).
    Partial credit is given for partially correct answers.
    """
    score: float = Field(description="Overall score 0.0 to 1.0")
    breakdown: Dict[str, float] = Field(description="Per-component scores showing what was right or wrong")
    feedback: str = Field(description="Human-readable explanation of the score")
    max_possible: float = Field(default=1.0, description="Maximum achievable score")