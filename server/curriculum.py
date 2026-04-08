"""
SYNAPSE — Curriculum Learning Manager
=======================================
Automatically adjusts scenario difficulty based on agent performance.

This is a KEY differentiator. Most environments give random scenarios.
SYNAPSE tracks agent performance and:
  - Starts with easy scenarios
  - Promotes to medium when agent consistently scores > 0.7
  - Promotes to hard when agent consistently scores > 0.7 on medium
  - Demotes if agent is struggling

This mirrors real-world ML curriculum learning (Bengio et al. 2009)
and enables faster, more efficient agent training.

Used by /train endpoint for multi-episode training loops.
"""

from typing import Dict, List, Optional
from collections import deque
import statistics


class CurriculumManager:
    """
    Tracks agent performance and manages difficulty progression.

    Difficulty levels:
        0 = easy   (nan_loss, gpu_oom, gradient_explosion)
        1 = medium (overfitting, underfitting, bad_deployment, memory_leak)
        2 = hard   (data_drift, class_imbalance, context_overflow, etc.)

    Promotion rule: Average score > 0.70 over last N episodes → promote
    Demotion rule:  Average score < 0.40 over last N episodes → demote
    """

    LEVELS = {
        0: "easy",
        1: "medium",
        2: "hard",
    }

    PROMOTE_THRESHOLD = 0.70   # Score needed to advance to next difficulty
    DEMOTE_THRESHOLD  = 0.40   # Score below which we drop back
    WINDOW_SIZE       = 5      # Episodes to average before level change

    def __init__(self):
        self.current_level: int = 0           # Start on easy
        self.episode_count: int = 0
        self.total_score:   float = 0.0

        # Sliding window of recent scores per difficulty
        self._windows: Dict[int, deque] = {
            0: deque(maxlen=self.WINDOW_SIZE),
            1: deque(maxlen=self.WINDOW_SIZE),
            2: deque(maxlen=self.WINDOW_SIZE),
        }

        # Full history for analytics
        self._history: List[Dict] = []

    def record(self, score: float, task_id: str, scenario_difficulty: str) -> Dict:
        """
        Record a completed episode score and decide if difficulty changes.

        Args:
            score:               Reward score 0.0 - 1.0
            task_id:             Which task was attempted
            scenario_difficulty: Actual scenario difficulty (easy/medium/hard)

        Returns:
            Dict with new difficulty level and promotion/demotion info
        """
        self.episode_count += 1
        self.total_score += score

        level_num = {"easy": 0, "medium": 1, "hard": 2}.get(scenario_difficulty, 0)
        self._windows[level_num].append(score)

        event = "none"
        old_level = self.current_level

        # Check promotion
        if (self.current_level < 2
                and len(self._windows[self.current_level]) >= self.WINDOW_SIZE):
            avg = statistics.mean(self._windows[self.current_level])
            if avg >= self.PROMOTE_THRESHOLD:
                self.current_level = min(2, self.current_level + 1)
                event = "promoted"
                # Clear window for new level
                self._windows[self.current_level] = deque(maxlen=self.WINDOW_SIZE)

        # Check demotion
        elif (self.current_level > 0
              and len(self._windows[self.current_level]) >= self.WINDOW_SIZE):
            avg = statistics.mean(self._windows[self.current_level])
            if avg < self.DEMOTE_THRESHOLD:
                self.current_level = max(0, self.current_level - 1)
                event = "demoted"

        record = {
            "episode":              self.episode_count,
            "score":                score,
            "task_id":              task_id,
            "scenario_difficulty":  scenario_difficulty,
            "current_difficulty":   self.LEVELS[self.current_level],
            "event":                event,
        }
        self._history.append(record)

        return {
            "level_changed": event != "none",
            "event":         event,
            "old_difficulty": self.LEVELS[old_level],
            "new_difficulty": self.LEVELS[self.current_level],
            "window_avg":    round(statistics.mean(self._windows[level_num]), 4)
                             if self._windows[level_num] else 0.0,
        }

    @property
    def current_difficulty(self) -> str:
        return self.LEVELS[self.current_level]

    def get_stats(self) -> Dict:
        """Return full curriculum statistics for /analytics endpoint."""
        if not self._history:
            return {
                "current_difficulty": self.current_difficulty,
                "episode_count": 0,
                "average_score": 0.0,
                "recent_scores": [],
                "learning_curve": [],
                "per_task_stats": {},
                "per_difficulty_stats": {},
            }

        scores = [h["score"] for h in self._history]
        avg = round(statistics.mean(scores), 4) if scores else 0.0

        # Per-task averages
        per_task: Dict[str, List[float]] = {}
        for h in self._history:
            per_task.setdefault(h["task_id"], []).append(h["score"])
        per_task_stats = {
            tid: {
                "count":   len(sc),
                "avg":     round(statistics.mean(sc), 4),
                "best":    round(max(sc), 4),
                "worst":   round(min(sc), 4),
            }
            for tid, sc in per_task.items()
        }

        # Per-difficulty averages
        per_diff: Dict[str, List[float]] = {}
        for h in self._history:
            per_diff.setdefault(h["scenario_difficulty"], []).append(h["score"])
        per_diff_stats = {
            diff: {
                "count": len(sc),
                "avg":   round(statistics.mean(sc), 4),
            }
            for diff, sc in per_diff.items()
        }

        # Learning curve: avg score per 5-episode window
        window = 5
        curve = []
        for i in range(0, len(scores), window):
            chunk = scores[i:i + window]
            curve.append(round(statistics.mean(chunk), 4))

        # Recent 10 scores
        recent = [
            {"episode": h["episode"], "score": h["score"], "task_id": h["task_id"]}
            for h in self._history[-10:]
        ]

        # Promote/demote events
        events = [h for h in self._history if h.get("event") != "none"]

        return {
            "current_difficulty":   self.current_difficulty,
            "current_level":        self.current_level,
            "episode_count":        self.episode_count,
            "total_score":          round(self.total_score, 4),
            "average_score":        avg,
            "recent_scores":        recent,
            "learning_curve":       curve,
            "per_task_stats":       per_task_stats,
            "per_difficulty_stats": per_diff_stats,
            "promote_threshold":    self.PROMOTE_THRESHOLD,
            "demote_threshold":     self.DEMOTE_THRESHOLD,
            "window_size":          self.WINDOW_SIZE,
            "curriculum_events":    len([h for h in self._history if h.get("event") != "none"]),
        }

    def reset_stats(self):
        """Reset all statistics. Used for fresh training runs."""
        self.__init__()