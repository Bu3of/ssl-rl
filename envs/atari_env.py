"""Atari-100k environment wrapper, matching MinAtarEnv's interface (reset/step/
obs_shape/num_actions) so it's a drop-in replacement wherever the rest of the
pipeline (RainbowAgent, SPRHead) expects an env -- only the encoder needs to
change to match the larger observation.

Preprocessing follows the standard Atari-100k / Data-Efficient Rainbow
protocol: grayscale, resize to 84x84, max-pool the last 2 raw frames, 4-frame
action repeat, up to 30 no-op steps on reset, episodic-life termination
during training. Built on gymnasium's own `AtariPreprocessing` /
`FrameStackObservation` wrappers rather than reimplementing them, since
those are the standard, widely-verified implementations of this protocol.

Games with a FIRE action (Breakout included) need it pressed once after
every reset to actually launch the ball -- without this, an agent that
hasn't yet learned to press FIRE can get stuck in a life that never ends
(ball never launches, so it never falls, so no life is ever lost), which
silently produces a flat, always-zero training curve. `reset()` presses it
automatically when present, matching the standard `FireResetEnv` wrapper
from the original DQN/Rainbow codebases (gymnasium has no built-in
equivalent).

Requires Atari ROMs (`AutoROM --accept-license`) and `ale-py`.
"""
from __future__ import annotations

import ale_py
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)


class AtariEnv:
    def __init__(self, game_id: str, seed: int | None = None, frame_stack: int = 4, terminal_on_life_loss: bool = True):
        self.game_id = game_id
        env = gym.make(game_id, frameskip=1, repeat_action_probability=0.0)
        env = AtariPreprocessing(
            env, noop_max=30, frame_skip=4, screen_size=84,
            terminal_on_life_loss=terminal_on_life_loss, grayscale_obs=True, scale_obs=False,
        )
        env = FrameStackObservation(env, stack_size=frame_stack)
        self._env = env
        self._seed = seed
        self._num_actions = env.action_space.n
        meanings = env.unwrapped.get_action_meanings()
        self._fire_action = meanings.index("FIRE") if "FIRE" in meanings else None

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def obs_shape(self) -> tuple:
        return self._env.observation_space.shape  # (frame_stack, 84, 84)

    def reset(self) -> np.ndarray:
        obs, info = self._env.reset(seed=self._seed)
        if self._fire_action is not None:
            obs, _, terminated, truncated, info = self._env.step(self._fire_action)
            if terminated or truncated:  # dying immediately on fire is rare but possible; just retry
                return self.reset()
        return np.asarray(obs, dtype=np.float32) / 255.0

    def step(self, action_idx: int):
        obs, reward, terminated, truncated, info = self._env.step(action_idx)
        return np.asarray(obs, dtype=np.float32) / 255.0, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self._env.close()
