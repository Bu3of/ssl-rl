"""Thin wrapper around minatar.Environment exposing a gymnasium-style step/reset API.

MinAtar quirks this wrapper hides from the rest of the pipeline:
- `Environment.act(a)` always expects an index into the full 6-action map
  ('n','l','u','r','d','f'), even though most games only use a few of them.
  `minimal_action_set()` returns which of those 6 indices are meaningful for
  a given game, so the Q-network only needs to output that many logits; we
  translate its chosen index back to the raw action before calling `act`.
- `state()` returns a (10, 10, C) boolean grid; PyTorch conv layers want
  (C, H, W) float32.
- `reset()` mutates internal state and returns None, so the first
  observation has to be fetched with a separate `state()` call.
"""
from __future__ import annotations

import numpy as np
from minatar import Environment

GAMES = ("asterix", "breakout", "freeway", "seaquest", "space_invaders")


class MinAtarEnv:
    def __init__(self, game, sticky_action_prob=0.1, difficulty_ramping=True, seed=None):
        if game not in GAMES:
            raise ValueError(f"Unknown MinAtar game '{game}', expected one of {GAMES}")
        self.game = game
        self._env = Environment(game, sticky_action_prob=sticky_action_prob, difficulty_ramping=difficulty_ramping)
        if seed is not None:
            self._env.seed(seed)
        self._actions = self._env.minimal_action_set()

    @property
    def num_actions(self):
        return len(self._actions)

    @property
    def obs_shape(self):
        h, w, c = self._env.state_shape()
        return (c, h, w)

    def _obs(self):
        return np.asarray(self._env.state(), dtype=np.float32).transpose(2, 0, 1)

    def reset(self):
        self._env.reset()
        return self._obs()

    def step(self, action_idx):
        reward, terminal = self._env.act(self._actions[action_idx])
        return self._obs(), float(reward), bool(terminal), False, {}


def make_env(game, **kwargs):
    return MinAtarEnv(game, **kwargs)
