"""
SYNAPSE — Incident Scenarios
==============================
10 realistic AI model production incidents with authentic PyTorch data.

Each scenario contains:
- Real PyTorch training log formats (matching actual PyTorch output)
- Realistic production metrics (matching real monitoring systems)
- Authentic error messages (copied from actual PyTorch exceptions)
- Ground truth answers for all 5 tasks

Scenario catalogue:
  sc_001 — NaN Loss (Learning Rate Too High)
  sc_002 — CUDA Out of Memory (Batch Size Too Large)
  sc_003 — Overfitting in Production
  sc_004 — Gradient Explosion (No Gradient Clipping)
  sc_005 — Underfitting (Learning Rate Too Low)
  sc_006 — Data Drift (Distribution Shift)
  sc_007 — Bad Deployment (Wrong Model Version)
  sc_008 — Memory Leak in DataLoader
  sc_009 — Class Imbalance Causing Bias
  sc_010 — Context Window Overflow in LLM
"""

SCENARIOS = [

    # =========================================================================
    # SCENARIO 1 — NaN Loss: Learning Rate Too High
    # =========================================================================
    {
        "id": "sc_001",
        "name": "NaN Loss Explosion",
        "pytorch_version": "2.1.0",
        "cuda_version": "11.8",
        "model_config": {
            "architecture": "ResNet50",
            "optimizer": "Adam",
            "learning_rate": 0.1,          # Too high for Adam — causes NaN
            "batch_size": 64,
            "epochs": 10,
            "dropout": 0.3,
            "weight_decay": 0.0001,
            "gradient_clip": None,          # No clipping — makes it worse
            "scheduler": None,
            "pytorch_version": "2.1.0",
            "cuda_version": "11.8",
        },
        "training_logs": [
            {"epoch": 1, "total_epochs": 10, "train_loss": 2.891, "val_loss": 2.934,
             "train_acc": 23.4, "val_acc": 22.1, "learning_rate": 0.1,
             "timestamp": "10:00:01", "warning": None, "error": None},
            {"epoch": 2, "total_epochs": 10, "train_loss": 4.231, "val_loss": 4.891,
             "train_acc": 18.2, "val_acc": 17.9, "learning_rate": 0.1,
             "timestamp": "10:02:14", "warning": "Loss is increasing — possible instability detected", "error": None},
            {"epoch": 3, "total_epochs": 10, "train_loss": 9.847, "val_loss": 11.23,
             "train_acc": 10.1, "val_acc": 9.8, "learning_rate": 0.1,
             "timestamp": "10:04:28", "warning": "Gradient norm: 847.3 — exploding gradients", "error": None},
            {"epoch": 4, "total_epochs": 10, "train_loss": None, "val_loss": None,
             "train_acc": 0.0, "val_acc": 0.0, "learning_rate": 0.1,
             "timestamp": "10:06:41", "warning": None,
             "error": "Loss is nan at step 847. Training collapsed completely."},
        ],
        "production_metrics": {
            "requests_per_second": 0.0, "p99_latency_ms": 0.0,
            "error_rate": 1.0, "hallucination_rate": 0.0,
            "accuracy": 0.0, "drift_score": 0.0,
            "gpu_memory_used_gb": 4.2, "gpu_memory_total_gb": 16.0,
        },
        "error_traces": [{"error_type": "RuntimeError",
            "message": "Function 'AddmmBackward0' returned nan values in its 0th output. Often caused by overflow in model parameters. Try reducing the learning rate.",
            "file": "torch/nn/modules/linear.py", "line": 114, "pytorch_component": "optimizer"}],
        "deployment_history": [{"event_type": "config_change", "version": "v1.2.0",
            "timestamp": "09:55:00", "changed_by": "ml-engineer-01",
            "description": "Increased learning rate from 0.001 to 0.1 for faster convergence"}],
        "alert_summary": "Training crashed at epoch 4. Loss became NaN after learning rate was increased 100x. Model accuracy is 0%.",
        "hint": "Look at the learning_rate in model_config and compare with the gradient_norm warning in training logs.",
        "ground_truth": {
            "task1": {"anomaly_detected": True, "anomaly_type": "nan_loss", "severity": "critical"},
            "task2": {"root_cause": "learning_rate_too_high", "affected_components": ["optimizer", "loss_function"]},
            "task3": {"priority_ranking": ["resnet50-prod"], "response_team": "ml_team"},
            "task4": {
                "immediate_steps": ["stop training immediately", "revert learning_rate to 0.001"],
                "root_fix_steps": ["add gradient_clip max_norm=1.0", "use lr_scheduler with warmup", "restart from last valid checkpoint"],
                "verification_steps": ["verify loss decreasing for 3 epochs", "verify gradient norm below 10.0", "verify val_acc improving"],
            },
            "task5": {
                "root_cause": "learning_rate_too_high",
                "affected_components": ["optimizer", "loss_function"],
                "prevention_steps": ["add_lr_scheduler", "add_gradient_clipping"],
                "monitoring_additions": ["nan_loss_alert", "gradient_norm_monitor"],
            },
        },
    },

    # =========================================================================
    # SCENARIO 2 — CUDA Out of Memory: Batch Size Too Large
    # =========================================================================
    {
        "id": "sc_002",
        "name": "CUDA Out of Memory",
        "pytorch_version": "2.1.0",
        "cuda_version": "11.8",
        "model_config": {
            "architecture": "BERT-Large",
            "optimizer": "AdamW",
            "learning_rate": 0.00002,
            "batch_size": 128,             # Too large for 16GB GPU with BERT-Large
            "epochs": 5,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "scheduler": "linear_warmup",
            "pytorch_version": "2.1.0",
            "cuda_version": "11.8",
        },
        "training_logs": [
            {"epoch": 1, "total_epochs": 5, "train_loss": 0.891, "val_loss": 0.923,
             "train_acc": 71.2, "val_acc": 69.8, "learning_rate": 0.00002,
             "timestamp": "14:00:01", "warning": "GPU memory at 89% — consider reducing batch_size", "error": None},
            {"epoch": 2, "total_epochs": 5, "train_loss": 0.743, "val_loss": 0.781,
             "train_acc": 76.4, "val_acc": 74.1, "learning_rate": 0.00002,
             "timestamp": "14:08:33", "warning": "GPU memory at 94% — approaching limit", "error": None},
            {"epoch": 3, "total_epochs": 5, "train_loss": None, "val_loss": None,
             "train_acc": None, "val_acc": None, "learning_rate": 0.00002,
             "timestamp": "14:17:12", "warning": None,
             "error": "CUDA out of memory. Tried to allocate 2.50 GiB. GPU 0 has 16.00 GiB total capacity; 15.12 GiB already allocated; 1.23 GiB free."},
        ],
        "production_metrics": {
            "requests_per_second": 0.0, "p99_latency_ms": 0.0,
            "error_rate": 1.0, "hallucination_rate": 0.0,
            "accuracy": 0.0, "drift_score": 0.0,
            "gpu_memory_used_gb": 15.98, "gpu_memory_total_gb": 16.0,
        },
        "error_traces": [{"error_type": "RuntimeError",
            "message": "CUDA out of memory. Tried to allocate 2.50 GiB (GPU 0; 16.00 GiB total capacity; 15.12 GiB already allocated; 1.23 GiB free; 1.07 GiB cached)",
            "file": "torch/nn/modules/module.py", "line": 1501, "pytorch_component": "dataloader"}],
        "deployment_history": [{"event_type": "config_change", "version": "v2.0.0",
            "timestamp": "13:55:00", "changed_by": "ml-engineer-02",
            "description": "Increased batch_size from 32 to 128 to speed up training"}],
        "alert_summary": "BERT-Large training crashed at epoch 3 with CUDA OOM. GPU memory 99.9% utilized after batch_size was quadrupled.",
        "hint": "Check gpu_memory_used_gb vs gpu_memory_total_gb in production_metrics and the recent config change.",
        "ground_truth": {
            "task1": {"anomaly_detected": True, "anomaly_type": "gpu_oom", "severity": "critical"},
            "task2": {"root_cause": "memory_leak", "affected_components": ["dataloader", "model"]},
            "task3": {"priority_ranking": ["bert-large-prod"], "response_team": "ml_team"},
            "task4": {
                "immediate_steps": ["stop training", "reduce batch_size to 32"],
                "root_fix_steps": ["enable gradient_checkpointing", "use mixed_precision fp16", "restart training"],
                "verification_steps": ["verify GPU memory below 80%", "verify training completes epoch 3", "verify val_acc improving"],
            },
            "task5": {
                "root_cause": "memory_leak",
                "affected_components": ["dataloader", "model"],
                "prevention_steps": ["add_memory_profiler", "validate_batch_size_before_training"],
                "monitoring_additions": ["gpu_memory_alert_80pct", "oom_predictor"],
            },
        },
    },

    # =========================================================================
    # SCENARIO 3 — Overfitting in Production
    # =========================================================================
    {
        "id": "sc_003",
        "name": "Overfitting in Production",
        "pytorch_version": "2.1.0",
        "cuda_version": "12.1",
        "model_config": {
            "architecture": "GPT2-Medium",
            "optimizer": "Adam",
            "learning_rate": 0.0003,
            "batch_size": 32,
            "epochs": 50,
            "dropout": 0.0,                # No dropout — causes overfitting
            "weight_decay": 0.0,            # No regularization
            "gradient_clip": 1.0,
            "scheduler": None,
            "pytorch_version": "2.1.0",
            "cuda_version": "12.1",
        },
        "training_logs": [
            {"epoch": 10, "total_epochs": 50, "train_loss": 2.341, "val_loss": 2.389,
             "train_acc": 54.2, "val_acc": 53.1, "learning_rate": 0.0003,
             "timestamp": "09:10:00", "warning": None, "error": None},
            {"epoch": 20, "total_epochs": 50, "train_loss": 1.823, "val_loss": 2.156,
             "train_acc": 67.8, "val_acc": 61.2, "learning_rate": 0.0003,
             "timestamp": "09:30:00", "warning": "val_loss increasing while train_loss decreasing — overfitting signal", "error": None},
            {"epoch": 30, "total_epochs": 50, "train_loss": 1.124, "val_loss": 2.891,
             "train_acc": 79.3, "val_acc": 54.8, "learning_rate": 0.0003,
             "timestamp": "09:50:00", "warning": "Overfitting detected: train/val gap = 24.5%", "error": None},
            {"epoch": 40, "total_epochs": 50, "train_loss": 0.743, "val_loss": 3.891,
             "train_acc": 89.1, "val_acc": 42.3, "learning_rate": 0.0003,
             "timestamp": "10:10:00", "warning": "SEVERE overfitting: train/val accuracy gap = 46.8%", "error": None},
        ],
        "production_metrics": {
            "requests_per_second": 340.0, "p99_latency_ms": 280.0,
            "error_rate": 0.02, "hallucination_rate": 0.38,
            "accuracy": 0.42, "drift_score": 0.71,
            "gpu_memory_used_gb": 8.4, "gpu_memory_total_gb": 24.0,
        },
        "error_traces": [],
        "deployment_history": [{"event_type": "deploy", "version": "v3.0.0",
            "timestamp": "08:00:00", "changed_by": "ml-engineer-03",
            "description": "Deployed GPT2-Medium trained for 40 epochs without dropout or weight_decay"}],
        "alert_summary": "Production model accuracy dropped to 42%. Hallucination rate at 38%. Model memorized training data but fails on new inputs.",
        "hint": "Compare train_acc vs val_acc across epochs. Check dropout and weight_decay in model_config.",
        "ground_truth": {
            "task1": {"anomaly_detected": True, "anomaly_type": "accuracy_drop", "severity": "high"},
            "task2": {"root_cause": "overfitting", "affected_components": ["model", "loss_function"]},
            "task3": {"priority_ranking": ["gpt2-medium-prod"], "response_team": "ml_team"},
            "task4": {
                "immediate_steps": ["rollback to previous model version", "disable current deployment"],
                "root_fix_steps": ["add dropout=0.3", "add weight_decay=0.01", "retrain with early_stopping"],
                "verification_steps": ["verify train/val gap below 5%", "verify production accuracy above 80%", "verify hallucination_rate below 10%"],
            },
            "task5": {
                "root_cause": "overfitting",
                "affected_components": ["model", "loss_function"],
                "prevention_steps": ["add_dropout", "add_weight_decay", "add_early_stopping"],
                "monitoring_additions": ["train_val_gap_monitor", "production_accuracy_alert"],
            },
        },
    },

    # =========================================================================
    # SCENARIO 4 — Gradient Explosion: No Gradient Clipping
    # =========================================================================
    {
        "id": "sc_004",
        "name": "Gradient Explosion",
        "pytorch_version": "2.2.0",
        "cuda_version": "12.1",
        "model_config": {
            "architecture": "LSTM-Large",
            "optimizer": "SGD",
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 20,
            "dropout": 0.2,
            "weight_decay": 0.0,
            "gradient_clip": None,          # Missing clipping — fatal for RNNs
            "scheduler": None,
            "pytorch_version": "2.2.0",
            "cuda_version": "12.1",
        },
        "training_logs": [
            {"epoch": 1, "total_epochs": 20, "train_loss": 3.21, "val_loss": 3.34,
             "train_acc": 31.2, "val_acc": 29.8, "learning_rate": 0.01,
             "timestamp": "08:00:01", "warning": None, "error": None},
            {"epoch": 2, "total_epochs": 20, "train_loss": 2.87, "val_loss": 2.94,
             "train_acc": 38.4, "val_acc": 37.1, "learning_rate": 0.01,
             "timestamp": "08:05:22", "warning": "Gradient norm: 124.3 — unusually high", "error": None},
            {"epoch": 3, "total_epochs": 20, "train_loss": 8941.2, "val_loss": None,
             "train_acc": 0.1, "val_acc": None, "learning_rate": 0.01,
             "timestamp": "08:10:44", "warning": None,
             "error": "Gradient norm is 98432.1. Model weights have exploded. Loss jumped from 2.87 to 8941.2"},
        ],
        "production_metrics": {
            "requests_per_second": 0.0, "p99_latency_ms": 0.0,
            "error_rate": 1.0, "hallucination_rate": 0.0,
            "accuracy": 0.01, "drift_score": 0.0,
            "gpu_memory_used_gb": 6.1, "gpu_memory_total_gb": 16.0,
        },
        "error_traces": [{"error_type": "RuntimeError",
            "message": "Gradient norm is 98432.12. Gradients have exploded. Add torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) before optimizer.step()",
            "file": "torch/nn/utils/clip_grad.py", "line": 42, "pytorch_component": "optimizer"}],
        "deployment_history": [{"event_type": "config_change", "version": "v1.1.0",
            "timestamp": "07:55:00", "changed_by": "ml-engineer-04",
            "description": "Removed gradient clipping to test if it was slowing convergence"}],
        "alert_summary": "LSTM training exploded at epoch 3. Gradient norm reached 98432. Loss jumped from 2.87 to 8941. Gradient clipping was removed yesterday.",
        "hint": "Check gradient_clip in model_config and read the error message carefully.",
        "ground_truth": {
            "task1": {"anomaly_detected": True, "anomaly_type": "gradient_explosion", "severity": "critical"},
            "task2": {"root_cause": "gradient_explosion", "affected_components": ["optimizer", "model"]},
            "task3": {"priority_ranking": ["lstm-large-prod"], "response_team": "ml_team"},
            "task4": {
                "immediate_steps": ["stop training immediately", "restore gradient_clip=1.0"],
                "root_fix_steps": ["add clip_grad_norm_ before optimizer.step", "restart training from epoch 1"],
                "verification_steps": ["verify gradient_norm below 10.0 each epoch", "verify loss decreasing steadily"],
            },
            "task5": {
                "root_cause": "gradient_explosion",
                "affected_components": ["optimizer", "model"],
                "prevention_steps": ["always_use_gradient_clipping_for_rnns", "monitor_gradient_norm_each_epoch"],
                "monitoring_additions": ["gradient_norm_alert_threshold_100", "loss_spike_detector"],
            },
        },
    },

    # =========================================================================
    # SCENARIO 5 — Underfitting: Learning Rate Too Low
    # =========================================================================
    {
        "id": "sc_005",
        "name": "Underfitting - Learning Rate Too Low",
        "pytorch_version": "2.1.0",
        "cuda_version": "11.8",
        "model_config": {
            "architecture": "ResNet18",
            "optimizer": "Adam",
            "learning_rate": 0.000001,      # Way too low — model barely learns
            "batch_size": 64,
            "epochs": 30,
            "dropout": 0.4,
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "scheduler": None,
            "pytorch_version": "2.1.0",
            "cuda_version": "11.8",
        },
        "training_logs": [
            {"epoch": 5,  "total_epochs": 30, "train_loss": 2.298, "val_loss": 2.301,
             "train_acc": 26.1, "val_acc": 25.9, "learning_rate": 0.000001,
             "timestamp": "11:10:00", "warning": None, "error": None},
            {"epoch": 10, "total_epochs": 30, "train_loss": 2.289, "val_loss": 2.292,
             "train_acc": 26.4, "val_acc": 26.2, "learning_rate": 0.000001,
             "timestamp": "11:30:00", "warning": "Model not converging — loss barely changing after 10 epochs", "error": None},
            {"epoch": 20, "total_epochs": 30, "train_loss": 2.271, "val_loss": 2.274,
             "train_acc": 26.9, "val_acc": 26.7, "learning_rate": 0.000001,
             "timestamp": "12:10:00", "warning": "UNDERFITTING: After 20 epochs accuracy still near random chance (25%)", "error": None},
            {"epoch": 30, "total_epochs": 30, "train_loss": 2.261, "val_loss": 2.265,
             "train_acc": 27.1, "val_acc": 26.9, "learning_rate": 0.000001,
             "timestamp": "13:00:00", "warning": "Training complete but model stuck near random baseline. Expected >85% accuracy.", "error": None},
        ],
        "production_metrics": {
            "requests_per_second": 210.0, "p99_latency_ms": 120.0,
            "error_rate": 0.01, "hallucination_rate": 0.0,
            "accuracy": 0.27, "drift_score": 0.08,
            "gpu_memory_used_gb": 3.2, "gpu_memory_total_gb": 16.0,
        },
        "error_traces": [],
        "deployment_history": [{"event_type": "config_change", "version": "v2.1.0",
            "timestamp": "09:00:00", "changed_by": "ml-engineer-05",
            "description": "Reduced learning_rate from 0.001 to 0.000001 after loss was oscillating"}],
        "alert_summary": "ResNet18 deployed with 27% accuracy — near random chance for 4-class classification. Model trained for 30 epochs but barely improved from epoch 1.",
        "hint": "Compare train_acc across all epochs. A well-trained ResNet18 should reach 85%+ on standard datasets.",
        "ground_truth": {
            "task1": {"anomaly_detected": True, "anomaly_type": "accuracy_drop", "severity": "high"},
            "task2": {"root_cause": "learning_rate_too_low", "affected_components": ["optimizer"]},
            "task3": {"priority_ranking": ["resnet18-prod"], "response_team": "ml_team"},
            "task4": {
                "immediate_steps": ["take model offline", "increase learning_rate to 0.001"],
                "root_fix_steps": ["retrain with lr=0.001", "add cosine_annealing scheduler", "train until val_acc > 80%"],
                "verification_steps": ["verify loss drops >50% in first 5 epochs", "verify val_acc above 80%"],
            },
            "task5": {
                "root_cause": "learning_rate_too_low",
                "affected_components": ["optimizer"],
                "prevention_steps": ["use_lr_finder_before_training", "set_convergence_threshold"],
                "monitoring_additions": ["accuracy_improvement_rate_monitor", "convergence_speed_alert"],
            },
        },
    },

    # =========================================================================
    # SCENARIO 6 — Data Drift: Distribution Shift
    # =========================================================================
    {
        "id": "sc_006",
        "name": "Data Drift - Distribution Shift",
        "pytorch_version": "2.1.0",
        "cuda_version": "12.1",
        "model_config": {
            "architecture": "DistilBERT",
            "optimizer": "AdamW",
            "learning_rate": 0.00003,
            "batch_size": 32,
            "epochs": 10,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "scheduler": "linear",
            "pytorch_version": "2.1.0",
            "cuda_version": "12.1",
        },
        "training_logs": [
            {"epoch": 10, "total_epochs": 10, "train_loss": 0.234, "val_loss": 0.241,
             "train_acc": 91.2, "val_acc": 90.8, "learning_rate": 0.00003,
             "timestamp": "2025-01-15 09:00:00", "warning": None, "error": None},
        ],
        "production_metrics": {
            "requests_per_second": 890.0, "p99_latency_ms": 95.0,
            "error_rate": 0.03, "hallucination_rate": 0.0,
            "accuracy": 0.61, "drift_score": 0.84,
            "gpu_memory_used_gb": 4.1, "gpu_memory_total_gb": 24.0,
        },
        "error_traces": [],
        "deployment_history": [
            {"event_type": "deploy", "version": "v1.0.0", "timestamp": "2025-01-15 10:00:00",
             "changed_by": "ml-engineer-06", "description": "Initial deployment — trained on Q4 2024 customer reviews"},
            {"event_type": "scale", "version": "v1.0.0", "timestamp": "2025-03-01 00:00:00",
             "changed_by": "auto-scaler", "description": "New product launch — 10x traffic increase from new customer segment"},
        ],
        "alert_summary": "Sentiment model accuracy dropped from 91% to 61% over 6 weeks. Drift score is 0.84 (critical). Model was trained on old customer data but new product users write very differently.",
        "hint": "Look at when the accuracy started dropping vs the deployment history events.",
        "ground_truth": {
            "task1": {"anomaly_detected": True, "anomaly_type": "data_drift", "severity": "high"},
            "task2": {"root_cause": "data_drift", "affected_components": ["model"]},
            "task3": {"priority_ranking": ["distilbert-sentiment-prod"], "response_team": "ml_team"},
            "task4": {
                "immediate_steps": ["collect new customer segment data samples", "measure drift across input features"],
                "root_fix_steps": ["retrain model on new customer segment data", "implement continuous learning pipeline"],
                "verification_steps": ["verify drift_score below 0.2", "verify accuracy above 85% on new data"],
            },
            "task5": {
                "root_cause": "data_drift",
                "affected_components": ["model"],
                "prevention_steps": ["add_drift_detection_pipeline", "schedule_regular_retraining"],
                "monitoring_additions": ["drift_score_alert_0.3", "accuracy_degradation_alert"],
            },
        },
    },

    # =========================================================================
    # SCENARIO 7 — Bad Deployment: Wrong Model Version
    # =========================================================================
    {
        "id": "sc_007",
        "name": "Bad Deployment - Wrong Model Version",
        "pytorch_version": "2.1.0",
        "cuda_version": "12.1",
        "model_config": {
            "architecture": "EfficientNet-B4",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 25,
            "dropout": 0.3,
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "scheduler": "cosine_annealing",
            "pytorch_version": "2.1.0",
            "cuda_version": "12.1",
        },
        "training_logs": [
            {"epoch": 25, "total_epochs": 25, "train_loss": 0.187, "val_loss": 0.201,
             "train_acc": 94.8, "val_acc": 93.2, "learning_rate": 0.0001,
             "timestamp": "15:00:00", "warning": None, "error": None},
        ],
        "production_metrics": {
            "requests_per_second": 450.0, "p99_latency_ms": 340.0,
            "error_rate": 0.43, "hallucination_rate": 0.0,
            "accuracy": 0.34, "drift_score": 0.12,
            "gpu_memory_used_gb": 9.8, "gpu_memory_total_gb": 24.0,
        },
        "error_traces": [{"error_type": "RuntimeError",
            "message": "Error(s) in loading state_dict for EfficientNet: Missing key(s): classifier.1.weight, classifier.1.bias. Unexpected key(s): classifier.weight, classifier.bias",
            "file": "torch/nn/modules/module.py", "line": 2041, "pytorch_component": "model"}],
        "deployment_history": [
            {"event_type": "deploy", "version": "v4.2.0", "timestamp": "11:30:00",
             "changed_by": "ml-engineer-07", "description": "Deployed v4.2.0 model weights"},
            {"event_type": "deploy", "version": "v4.2.0", "timestamp": "14:55:00",
             "changed_by": "ci-cd-pipeline", "description": "Emergency hotfix — accidentally deployed v3.1.0 weights to v4.2.0 container. Architecture mismatch."},
        ],
        "alert_summary": "EfficientNet classifier error rate jumped to 43% after hotfix deployment. State dict keys don't match — wrong model weights deployed to wrong architecture container.",
        "hint": "Read the error trace carefully and look at the last deployment event description.",
        "ground_truth": {
            "task1": {"anomaly_detected": True, "anomaly_type": "accuracy_drop", "severity": "critical"},
            "task2": {"root_cause": "bad_deployment", "affected_components": ["model"]},
            "task3": {"priority_ranking": ["efficientnet-b4-prod"], "response_team": "devops_team"},
            "task4": {
                "immediate_steps": ["rollback to v4.1.0 immediately", "disable ci-cd auto-deployment"],
                "root_fix_steps": ["verify model weights match architecture before deployment", "add state_dict validation step to CI/CD"],
                "verification_steps": ["verify error_rate below 5%", "verify accuracy above 90%"],
            },
            "task5": {
                "root_cause": "bad_deployment",
                "affected_components": ["model"],
                "prevention_steps": ["add_model_architecture_validation", "add_canary_deployment"],
                "monitoring_additions": ["error_rate_spike_alert", "model_version_mismatch_detector"],
            },
        },
    },

    # =========================================================================
    # SCENARIO 8 — Memory Leak in DataLoader
    # =========================================================================
    {
        "id": "sc_008",
        "name": "Memory Leak in DataLoader",
        "pytorch_version": "2.1.0",
        "cuda_version": "11.8",
        "model_config": {
            "architecture": "ResNet101",
            "optimizer": "SGD",
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 100,
            "dropout": 0.2,
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "scheduler": "step_lr",
            "pytorch_version": "2.1.0",
            "cuda_version": "11.8",
        },
        "training_logs": [
            {"epoch": 10,  "total_epochs": 100, "train_loss": 1.234, "val_loss": 1.287,
             "train_acc": 62.1, "val_acc": 60.8, "learning_rate": 0.01,
             "timestamp": "08:30:00", "warning": "GPU memory: 6.2 GB / 16.0 GB", "error": None},
            {"epoch": 30,  "total_epochs": 100, "train_loss": 0.891, "val_loss": 0.934,
             "train_acc": 71.4, "val_acc": 69.9, "learning_rate": 0.01,
             "timestamp": "10:30:00", "warning": "GPU memory: 11.8 GB / 16.0 GB — memory growing each epoch", "error": None},
            {"epoch": 50,  "total_epochs": 100, "train_loss": 0.743, "val_loss": 0.791,
             "train_acc": 77.2, "val_acc": 75.8, "learning_rate": 0.001,
             "timestamp": "12:30:00", "warning": "GPU memory: 15.1 GB / 16.0 GB — CRITICAL: leak detected", "error": None},
            {"epoch": 55,  "total_epochs": 100, "train_loss": None, "val_loss": None,
             "train_acc": None, "val_acc": None, "learning_rate": 0.001,
             "timestamp": "13:00:00", "warning": None,
             "error": "CUDA out of memory at epoch 55. Memory grew from 6.2GB to 16.0GB over 55 epochs. DataLoader worker processes not releasing GPU tensors."},
        ],
        "production_metrics": {
            "requests_per_second": 0.0, "p99_latency_ms": 0.0,
            "error_rate": 1.0, "hallucination_rate": 0.0,
            "accuracy": 0.0, "drift_score": 0.0,
            "gpu_memory_used_gb": 16.0, "gpu_memory_total_gb": 16.0,
        },
        "error_traces": [{"error_type": "RuntimeError",
            "message": "CUDA out of memory at epoch 55. GPU memory grew steadily from 6.2GB to 16.0GB. Cause: DataLoader workers holding references to GPU tensors — move augmentation transforms to CPU.",
            "file": "torch/utils/data/dataloader.py", "line": 628, "pytorch_component": "dataloader"}],
        "deployment_history": [{"event_type": "config_change", "version": "v5.0.0",
            "timestamp": "08:00:00", "changed_by": "ml-engineer-08",
            "description": "Moved image augmentation transforms from CPU to GPU for speed — num_workers=8"}],
        "alert_summary": "ResNet101 OOM at epoch 55/100. Memory grew steadily from 6.2GB to 16GB over training. GPU augmentation in DataLoader workers is causing memory leak.",
        "hint": "GPU memory grew STEADILY each epoch — this is different from sudden OOM. Look at what changed in deployment history.",
        "ground_truth": {
            "task1": {"anomaly_detected": True, "anomaly_type": "memory_leak", "severity": "critical"},
            "task2": {"root_cause": "memory_leak", "affected_components": ["dataloader"]},
            "task3": {"priority_ranking": ["resnet101-prod"], "response_team": "ml_team"},
            "task4": {
                "immediate_steps": ["stop training", "move augmentation transforms back to CPU"],
                "root_fix_steps": ["set pin_memory=False in DataLoader", "add del tensor after each batch", "restart training from epoch 50 checkpoint"],
                "verification_steps": ["verify GPU memory stays constant across epochs", "verify training completes epoch 100"],
            },
            "task5": {
                "root_cause": "memory_leak",
                "affected_components": ["dataloader"],
                "prevention_steps": ["always_run_augmentation_on_cpu", "monitor_gpu_memory_trend"],
                "monitoring_additions": ["gpu_memory_growth_rate_alert", "epoch_memory_delta_monitor"],
            },
        },
    },

    # =========================================================================
    # SCENARIO 9 — Class Imbalance Causing Model Bias
    # =========================================================================
    {
        "id": "sc_009",
        "name": "Class Imbalance - Biased Model",
        "pytorch_version": "2.2.0",
        "cuda_version": "12.1",
        "model_config": {
            "architecture": "MobileNetV3",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 128,
            "epochs": 20,
            "dropout": 0.2,
            "weight_decay": 0.0,
            "gradient_clip": None,
            "scheduler": None,
            "pytorch_version": "2.2.0",
            "cuda_version": "12.1",
        },
        "training_logs": [
            {"epoch": 5,  "total_epochs": 20, "train_loss": 0.123, "val_loss": 0.131,
             "train_acc": 96.1, "val_acc": 95.8, "learning_rate": 0.001,
             "timestamp": "09:10:00", "warning": "WARNING: High accuracy but model predicts class_0 for 97% of samples", "error": None},
            {"epoch": 10, "total_epochs": 20, "train_loss": 0.098, "val_loss": 0.101,
             "train_acc": 97.2, "val_acc": 96.9, "learning_rate": 0.001,
             "timestamp": "09:30:00", "warning": "Class distribution: class_0=96%, class_1=3%, class_2=1%. Model ignoring minority classes.", "error": None},
            {"epoch": 20, "total_epochs": 20, "train_loss": 0.071, "val_loss": 0.074,
             "train_acc": 98.1, "val_acc": 97.8, "learning_rate": 0.001,
             "timestamp": "10:10:00", "warning": "F1-score for class_1: 0.02. F1-score for class_2: 0.00. Model useless for minority classes.", "error": None},
        ],
        "production_metrics": {
            "requests_per_second": 1200.0, "p99_latency_ms": 45.0,
            "error_rate": 0.02, "hallucination_rate": 0.0,
            "accuracy": 0.97, "drift_score": 0.05,
            "gpu_memory_used_gb": 2.1, "gpu_memory_total_gb": 16.0,
        },
        "error_traces": [],
        "deployment_history": [{"event_type": "deploy", "version": "v1.0.0",
            "timestamp": "08:00:00", "changed_by": "ml-engineer-09",
            "description": "Deployed fraud detection model — trained on imbalanced dataset (96% normal, 3% suspicious, 1% fraud)"}],
        "alert_summary": "Fraud detection model shows 97% accuracy but misses 98% of actual fraud cases. Class imbalance in training data caused model to always predict 'normal'.",
        "hint": "The overall accuracy looks great at 97% but check the class distribution warning in the training logs.",
        "ground_truth": {
            "task1": {"anomaly_detected": True, "anomaly_type": "accuracy_drop", "severity": "critical"},
            "task2": {"root_cause": "class_imbalance", "affected_components": ["model", "loss_function"]},
            "task3": {"priority_ranking": ["mobilenet-fraud-prod"], "response_team": "ml_team"},
            "task4": {
                "immediate_steps": ["take model offline", "alert stakeholders about false sense of accuracy"],
                "root_fix_steps": ["use weighted_cross_entropy_loss with class weights", "oversample minority classes with SMOTE", "retrain and evaluate with F1-score not accuracy"],
                "verification_steps": ["verify F1-score for fraud class above 0.7", "verify recall for fraud class above 0.8"],
            },
            "task5": {
                "root_cause": "class_imbalance",
                "affected_components": ["model", "loss_function"],
                "prevention_steps": ["always_check_class_distribution", "use_f1_not_accuracy_for_imbalanced"],
                "monitoring_additions": ["per_class_f1_monitor", "minority_class_recall_alert"],
            },
        },
    },

    # =========================================================================
    # SCENARIO 10 — Context Window Overflow in Production LLM
    # =========================================================================
    {
        "id": "sc_010",
        "name": "Context Window Overflow in LLM",
        "pytorch_version": "2.1.0",
        "cuda_version": "12.1",
        "model_config": {
            "architecture": "LLaMA-7B",
            "optimizer": "AdamW",
            "learning_rate": 0.00002,
            "batch_size": 8,
            "epochs": 3,
            "dropout": 0.1,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "scheduler": "cosine",
            "pytorch_version": "2.1.0",
            "cuda_version": "12.1",
        },
        "training_logs": [
            {"epoch": 3, "total_epochs": 3, "train_loss": 1.234, "val_loss": 1.267,
             "train_acc": None, "val_acc": None, "learning_rate": 0.000005,
             "timestamp": "2025-02-01 12:00:00", "warning": None, "error": None},
        ],
        "production_metrics": {
            "requests_per_second": 23.0, "p99_latency_ms": 8900.0,
            "error_rate": 0.38, "hallucination_rate": 0.67,
            "accuracy": 0.41, "drift_score": 0.15,
            "gpu_memory_used_gb": 14.9, "gpu_memory_total_gb": 24.0,
        },
        "error_traces": [{"error_type": "RuntimeError",
            "message": "Sequence length 5847 exceeds maximum context length 4096 for model LLaMA-7B. Token overflow causes attention mechanism to fail and produce garbage output.",
            "file": "transformers/modeling_llama.py", "line": 891, "pytorch_component": "model"}],
        "deployment_history": [
            {"event_type": "deploy", "version": "v2.0.0", "timestamp": "2025-02-01 14:00:00",
             "changed_by": "ml-engineer-10", "description": "Deployed LLaMA-7B fine-tuned assistant"},
            {"event_type": "config_change", "version": "v2.0.0", "timestamp": "2025-03-15 09:00:00",
             "changed_by": "product-team", "description": "Enabled long document summarization feature — users now uploading entire PDFs as context"},
        ],
        "alert_summary": "LLaMA-7B hallucination rate at 67%, error rate 38%, latency 8.9 seconds. Users are uploading long PDFs that exceed the 4096 token context window causing attention overflow.",
        "hint": "Check the error message for sequence length. Check what feature was enabled in the latest deployment event.",
        "ground_truth": {
            "task1": {"anomaly_detected": True, "anomaly_type": "hallucination_spike", "severity": "critical"},
            "task2": {"root_cause": "context_overflow", "affected_components": ["model"]},
            "task3": {"priority_ranking": ["llama-7b-prod"], "response_team": "ai_team"},
            "task4": {
                "immediate_steps": ["disable long document feature", "add input length validation max 3500 tokens"],
                "root_fix_steps": ["implement sliding window chunking for long documents", "upgrade to LLaMA-7B with 8192 context window"],
                "verification_steps": ["verify no requests exceed 4096 tokens", "verify hallucination_rate below 10%", "verify p99_latency below 2000ms"],
            },
            "task5": {
                "root_cause": "context_overflow",
                "affected_components": ["model"],
                "prevention_steps": ["add_input_length_validation", "implement_chunking_for_long_inputs"],
                "monitoring_additions": ["token_count_p99_alert", "context_overflow_error_counter"],
            },
        },
    },
]