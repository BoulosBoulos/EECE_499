"""Ring-buffered collision trajectory logger for PDE training and eval."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List


TRAJECTORY_COLUMNS = [
    "step",
    "ego_x",
    "ego_y",
    "ego_psi",
    "ego_v",
    "ego_a",
    "action",
    "reward",
    "min_ttc",
    "n_agents",
    "collision_agent_id",
    "terminal_flag",
]


class TrajectoryLogger:
    """Logs the last N collision episodes as CSV files in a ring buffer.

    Ring-buffer slot index is `episode_counter % max_episodes`, so once we have
    written `max_episodes` files the oldest slots get overwritten in order.
    """

    def __init__(self, output_dir: str, max_episodes: int = 50):
        self.output_dir = Path(output_dir) / "trajectories"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_episodes = int(max_episodes)
        self.episode_counter = 0
        self._slot_index = 0

    def log_collision_episode(
        self,
        steps: List[Dict[str, Any]],
        scenario: str,
        ego_maneuver: str,
        seed: int,
        episode_idx: int,
        terminal_step: int,
        collision_agent_id: str,
    ) -> str:
        slot = self._slot_index % self.max_episodes
        path = self.output_dir / f"collision_{slot:04d}.csv"

        header_comment = (
            f"# scenario={scenario}, ego_maneuver={ego_maneuver}, "
            f"seed={seed}, episode_idx={episode_idx}, "
            f"terminal_step={terminal_step}, collision_agent_id={collision_agent_id}\n"
        )
        with open(path, "w", newline="") as f:
            f.write(header_comment)
            writer = csv.DictWriter(f, fieldnames=TRAJECTORY_COLUMNS)
            writer.writeheader()
            for s in steps:
                row = {col: s.get(col, "") for col in TRAJECTORY_COLUMNS}
                writer.writerow(row)

        self._slot_index += 1
        self.episode_counter += 1
        return str(path)

    def n_logged(self) -> int:
        return self.episode_counter

    def n_on_disk(self) -> int:
        return min(self.episode_counter, self.max_episodes)
