"""Stores raw (non n-step-folded) transitions for SPR's multi-step latent
prediction loss.

The main `PrioritizedReplayBuffer` folds each transition into an n-step
(obs, next_obs) pair for the Q-learning bootstrap target, which throws away
the intermediate frames SPR needs to build a K-step prediction chain. This
buffer keeps every raw (obs, action, done) step instead and samples
contiguous K+1-length windows, independently and uniformly from the
prioritized buffer's own sampling.
"""
from __future__ import annotations

import numpy as np


class SequenceReplayBuffer:
    def __init__(self, capacity, obs_shape, sequence_length):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(self, obs, action, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self):
        return self.size >= self.sequence_length + 2

    def sample(self, batch_size):
        k = self.sequence_length
        # windows are sampled without wraparound: a window starting near the
        # write pointer in a full circular buffer would otherwise stitch
        # together two unrelated points in time.
        effective_size = self.size if self.size < self.capacity else self.capacity
        max_start = effective_size - k - 1
        starts = np.random.randint(0, max_start, size=batch_size)

        window = starts[:, None] + np.arange(k + 1)[None, :]
        obs_seq = self.obs[window]  # (B, K+1, C, H, W)

        action_window = starts[:, None] + np.arange(k)[None, :]
        actions_seq = self.actions[action_window]  # (B, K)
        dones_seq = self.dones[action_window]  # (B, K)

        # mask[:, j] = 1 iff no episode boundary occurred in steps 0..j, i.e.
        # predicting obs at position j+1 from obs at position 0 is still
        # within the same episode. Once a done is hit, every later step in
        # the window belongs to an unrelated reset episode and is masked out.
        cumulative_done = np.cumsum(dones_seq, axis=1)
        mask = (cumulative_done == 0).astype(np.float32)

        return obs_seq.astype(np.float32), actions_seq.astype(np.int64), mask
