"""SUMO T-intersection Gymnasium environment."""

# TODO: Implement SUMO env with discrete behaviors STOP/CREEP/YIELD/GO/ABORT,
# partial observability (occlusion), and physics-informed reward.
# Config: configs/scenario/, configs/reward/

try:
    import gymnasium as gym
    _Base = gym.Env
except ImportError:
    _Base = object  # fallback when gymnasium not installed


class SumoEnv(_Base):
    """Placeholder for SUMO T-intersection environment."""

    def __init__(self, config_path: str | None = None):
        super().__init__()
        # TODO: Load config, init TraCI, set obs/action spaces
        pass

    def reset(self, *, seed=None, options=None):
        # TODO: Reset SUMO, return obs, info
        return None, {}

    def step(self, action):
        # TODO: Execute action, return obs, reward, terminated, truncated, info
        return None, 0.0, False, False, {}
