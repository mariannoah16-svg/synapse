"""
SYNAPSE v5 Final — FastAPI Server
====================================
All OpenEnv required endpoints + visual dashboard + /web UI.

Standard OpenEnv:   /health /reset /step /state /tasks /grader /baseline
Advanced:           /train /analytics /generate /curriculum
Visual:             / (dashboard) and /web (OpenEnv web interface)
"""
import json, os, subprocess, sys
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from models import Action
from server.environment import SynapseEnvironment

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SYNAPSE",
    description=(
        "🧠 **SYNAPSE v5** — Systematic Neural Analysis and Production Supervision Environment\n\n"
        "The world's first OpenEnv environment with **multi-turn investigation**, "
        "**procedural generation** (12 failure modes × 10 architectures), "
        "and **curriculum learning**.\n\n"
        "**Domain:** AI model failures in production — the most expensive, "
        "most real, most unsolved problem in AI operations today.\n\n"
        "🔗 [Dashboard](/) | [API Docs](/docs) | [Tasks](/tasks)"
    ),
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment
env = SynapseEnvironment(seed=42, use_curriculum=True)

# ── Request schemas ───────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id:       Optional[str]  = None
    difficulty:    Optional[str]  = None
    seed:          Optional[int]  = None
    use_generated: bool           = True
    mode:          str            = "standard"

class StepRequest(BaseModel):
    action: Action

class TrainRequest(BaseModel):
    episodes: int           = 5
    task_id:  Optional[str] = None
    mode:     str           = "standard"

# ── Visual Dashboard ──────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    """SYNAPSE visual dashboard — served at root URL."""
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(
        content="<h1>🧠 SYNAPSE</h1><p><a href='/docs'>API Docs</a></p>"
    )

@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
def web_interface():
    """/web endpoint — same as / but at the OpenEnv standard path."""
    return dashboard()

# ── Standard OpenEnv Endpoints ────────────────────────────────────────────────

@app.get("/health", tags=["OpenEnv"])
def health():
    """Liveness probe. Must return 200 OK."""
    return {
        "status":      "ok",
        "environment": "SYNAPSE",
        "version":     "5.0.0",
        "features": [
            "multi_turn_investigation",
            "procedural_generation",
            "curriculum_learning",
            "12_failure_modes",
            "10_architectures",
            "5_standard_tasks",
            "visual_dashboard",
        ],
    }

@app.post("/reset", tags=["OpenEnv"])
def reset(req: ResetRequest = ResetRequest()):
    """Start a new episode. Returns initial observation."""
    global env
    if req.seed is not None:
        env = SynapseEnvironment(seed=req.seed, use_curriculum=True)
    obs = env.reset(
        task_id       = req.task_id,
        difficulty    = req.difficulty,
        use_generated = req.use_generated,
        mode          = req.mode,
    )
    return {
        "observation":          obs.model_dump(),
        "mode":                 req.mode,
        "curriculum_difficulty": (
            env._curriculum.current_difficulty if env._curriculum else None
        ),
    }

@app.post("/step", tags=["OpenEnv"])
def step(req: StepRequest):
    """Submit agent action. Returns observation, reward, done, info."""
    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "observation": obs.model_dump(),
        "reward":      reward.model_dump(),
        "done":        done,
        "info":        info,
    }

@app.get("/state", tags=["OpenEnv"])
def state():
    """Current environment state snapshot."""
    return env.state()

# ── Scoring Endpoints ─────────────────────────────────────────────────────────

@app.get("/tasks", tags=["Scoring"])
def tasks():
    """All tasks with action schemas and expected baseline scores."""
    return {"tasks": env.get_tasks()}

@app.get("/grader", tags=["Scoring"])
def grader():
    """Grader result for the last completed episode."""
    s = env.state()
    if not s["done"]:
        raise HTTPException(
            status_code=400,
            detail="Episode not complete. Call POST /step first."
        )
    return {
        "episode_id":          s["episode_id"],
        "task_id":             s["task_id"],
        "mode":                s["mode"],
        "scenario_id":         s["scenario_id"],
        "scenario_difficulty": s["scenario_difficulty"],
        "failure_mode":        s["failure_mode"],
        "last_reward":         s["last_reward"],
    }

@app.get("/baseline", tags=["Scoring"])
def baseline():
    """Run inference.py and return baseline scores for all 5 tasks."""
    try:
        result = subprocess.run(
            [sys.executable, "inference.py", "--mode", "api", "--seed", "42"],
            capture_output=True,
            text=True,
            timeout=1200,
            cwd=os.getcwd(),
        )
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"inference.py failed:\n{result.stderr}"
            )
        last_line = result.stdout.strip().split("\n")[-1]
        scores    = json.loads(last_line)
        return {"status": "success", "baseline_scores": scores}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="inference.py exceeded 20-minute limit")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Advanced Endpoints ────────────────────────────────────────────────────────

@app.post("/train", tags=["Advanced"])
def train(req: TrainRequest):
    """Run multi-episode training loop with curriculum. Returns learning stats."""
    if not 1 <= req.episodes <= 20:
        raise HTTPException(status_code=400, detail="episodes must be 1–20")

    tasks_list = (
        [req.task_id] * req.episodes if req.task_id
        else [f"task{(i % 5) + 1}" for i in range(req.episodes)]
    )
    results = []
    for i, tid in enumerate(tasks_list):
        env.reset(task_id=tid, use_generated=True, mode=req.mode)
        _, reward, _, info = env.step(Action())
        results.append({
            "episode":      i + 1,
            "task_id":      tid,
            "score":        reward.score,
            "difficulty":   info.get("scenario_difficulty"),
            "failure_mode": info.get("failure_mode"),
            "curriculum":   info.get("curriculum"),
        })

    scores = [r["score"] for r in results]
    avg    = round(sum(scores) / len(scores), 4)
    return {
        "episodes_run":  len(results),
        "average_score": avg,
        "best_score":    round(max(scores), 4),
        "worst_score":   round(min(scores), 4),
        "results":       results,
        "curriculum":    env._curriculum.get_stats() if env._curriculum else {},
    }

@app.get("/analytics", tags=["Advanced"])
def analytics():
    """Full learning analytics — learning curves, per-task stats, curriculum."""
    return env.get_analytics()

@app.get("/generate", tags=["Advanced"])
def generate(
    failure_mode: Optional[str] = None,
    difficulty:   Optional[str] = None,
    architecture: Optional[str] = None,
):
    """Preview a generated scenario (without ground truth answers)."""
    from server.generator import ScenarioGenerator, FAILURE_MODES, ARCHITECTURES
    gen      = ScenarioGenerator(seed=None)
    scenario = gen.generate(
        failure_mode=failure_mode,
        difficulty=difficulty,
        architecture=architecture,
    )
    public = {k: v for k, v in scenario.items() if k != "ground_truth"}
    public["available_failure_modes"]  = list(FAILURE_MODES.keys())
    public["available_architectures"]  = [a["name"] for a in ARCHITECTURES]
    return public

@app.get("/curriculum", tags=["Advanced"])
def curriculum():
    """Current curriculum difficulty level and progression history."""
    if not env._curriculum:
        return {"curriculum_enabled": False}
    return {"curriculum_enabled": True, **env._curriculum.get_stats()}

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    """Entry point for 'uv run server'. Must be named 'main' per OpenEnv spec."""
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


# Keep run as alias for backward compatibility
run = main

if __name__ == "__main__":
    main()
