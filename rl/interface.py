"""Pluggable RL algorithm interface: swap PPO, SAC, etc. via config."""

# TODO: Implement pluggable algo interface; load algorithm from configs/algo/


class AlgorithmInterface:
    """Placeholder for pluggable RL algorithm interface."""

    def __init__(self, env, config_path: str | None = None):
        # TODO: Load configs/algo/, instantiate SB3 or custom algo
        self.env = env
        pass

    def train(self, total_timesteps: int):
        """Run training."""
        # TODO: Implement
        pass

    def save(self, path: str):
        """Save model checkpoint."""
        # TODO: Implement
        pass
