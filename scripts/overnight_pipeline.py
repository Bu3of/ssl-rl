"""Runs the full overnight pipeline sequentially. Each stage is resume-safe
(skips work whose checkpoint already indicates completion) and prints
progress along the way. Results land in checkpoints/ and a JSON summary in
notebooks/_overnight_results.json, which notebooks/phase1_phase2_results.ipynb
reads to build its plots.

Plain script rather than a notebook deliberately: this runs for several
hours unattended, and a plain background process has fewer moving parts to
silently fail on than a long-lived Jupyter kernel does.
"""
import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # `python scripts/overnight_pipeline.py` puts scripts/ on sys.path, not the repo root

from envs.minatar_env import MinAtarEnv
from models.encoder import MinAtarEncoder
from models.spr_head import SPRHead
from models.curl_head import CURLHead
from models.iqrl_head import IQRLHead
from agents.rainbow_agent import RainbowAgent
from agents.sequence_buffer import SequenceReplayBuffer
from diagnostics.effective_rank import effective_rank, collect_embeddings
CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

GAME = "breakout"
NUM_STEPS = 100_000
LEARNING_STARTS = 1_000
LOG_EVERY = 5_000
CHECKPOINT_EVERY = 20_000
RANK_EVERY = 10_000
RANK_ROLLOUT_STEPS = 512


def train_minatar_run(run_name, seed, aux_kind=None, spr_momentum_tau=0.99, spr_aug_pad=2, num_steps=NUM_STEPS):
    """aux_kind: None (baseline), 'spr', 'curl', or 'iqrl'."""
    ckpt_path = os.path.join(CKPT_DIR, f"{run_name}.pt")

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = MinAtarEnv(GAME, seed=seed)
    eval_env = MinAtarEnv(GAME, seed=seed + 1000)
    encoder = MinAtarEncoder(in_channels=env.obs_shape[0])

    aux_head, seq_buffer = None, None
    if aux_kind == "spr":
        aux_head = SPRHead(encoder, num_actions=env.num_actions, sequence_length=5,
                            momentum_tau=spr_momentum_tau, aug_pad=spr_aug_pad)
        seq_buffer = SequenceReplayBuffer(capacity=100_000, obs_shape=env.obs_shape, sequence_length=5)
    elif aux_kind == "curl":
        aux_head = CURLHead(encoder, momentum_tau=0.99, aug_pad=2)
    elif aux_kind == "iqrl":
        aux_head = IQRLHead(encoder, num_actions=env.num_actions, sequence_length=5, fsq_dim=8, fsq_levels=5)
        seq_buffer = SequenceReplayBuffer(capacity=100_000, obs_shape=env.obs_shape, sequence_length=5)

    agent = RainbowAgent(
        encoder=encoder, num_actions=env.num_actions, obs_shape=env.obs_shape,
        device=torch.device("cpu"), aux_head=aux_head, sequence_buffer=seq_buffer,
        aux_batch_size=32, aux_loss_weight=1.0,
    )

    start_step = 0
    episode_returns, rank_history = [], []
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt.get("step", 0) >= num_steps:
            print(f"[{run_name}] already complete at step {ckpt['step']}, skipping")
            return {"episode_returns": ckpt.get("episode_returns", []), "rank_history": ckpt.get("rank_history", [])}
        agent.load_state_dict(ckpt["agent"])
        start_step = ckpt["step"]
        episode_returns = ckpt.get("episode_returns", [])
        rank_history = ckpt.get("rank_history", [])
        print(f"[{run_name}] resumed at step {start_step}")
    else:
        print(f"[{run_name}] starting from scratch (seed={seed}, aux={aux_kind}, tau={spr_momentum_tau}, pad={spr_aug_pad})")

    ep_return = 0.0
    obs = env.reset()
    t0 = time.time()

    for step in range(start_step, num_steps):
        action = np.random.randint(env.num_actions) if step < LEARNING_STARTS else agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.store_transition(obs, action, reward, next_obs, done)
        ep_return += reward
        obs = next_obs
        if done:
            episode_returns.append(ep_return)
            ep_return = 0.0
            obs = env.reset()
        if step >= LEARNING_STARTS:
            agent.train_step()

        if step % RANK_EVERY == 0 and step > start_step:
            emb = collect_embeddings(agent.online_net.encoder, agent, eval_env, n=RANK_ROLLOUT_STEPS)
            rank_history.append((step, effective_rank(emb)))

        if step % LOG_EVERY == 0 and episode_returns:
            elapsed = time.time() - t0
            print(f"[{run_name}] step={step} episodes={len(episode_returns)} "
                  f"mean_return(last20)={np.mean(episode_returns[-20:]):.2f} "
                  f"({elapsed:.0f}s elapsed)")

        if step % CHECKPOINT_EVERY == 0 and step > start_step:
            torch.save({"agent": agent.state_dict(), "step": step,
                        "episode_returns": episode_returns, "rank_history": rank_history}, ckpt_path)

    torch.save({"agent": agent.state_dict(), "step": num_steps,
                "episode_returns": episode_returns, "rank_history": rank_history}, ckpt_path)
    print(f"[{run_name}] DONE ({time.time()-t0:.0f}s)")
    return {"episode_returns": episode_returns, "rank_history": rank_history}


def train_atari_run(run_name, num_steps=50_000):
    from envs.atari_env import AtariEnv
    from models.atari_encoder import AtariEncoder

    ckpt_path = os.path.join(CKPT_DIR, f"{run_name}.pt")
    torch.manual_seed(0)
    np.random.seed(0)
    env = AtariEnv("ALE/Breakout-v5", seed=0)
    encoder = AtariEncoder(in_channels=env.obs_shape[0])
    agent = RainbowAgent(encoder=encoder, num_actions=env.num_actions, obs_shape=env.obs_shape,
                          device=torch.device("cpu"), buffer_capacity=50_000)

    start_step = 0
    episode_returns = []
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt.get("step", 0) >= num_steps:
            print(f"[{run_name}] already complete, skipping")
            return {"episode_returns": ckpt.get("episode_returns", [])}
        agent.load_state_dict(ckpt["agent"])
        start_step = ckpt["step"]
        episode_returns = ckpt.get("episode_returns", [])
        print(f"[{run_name}] resumed at step {start_step}")
    else:
        print(f"[{run_name}] starting from scratch")

    ep_return = 0.0
    obs = env.reset()
    t0 = time.time()
    for step in range(start_step, num_steps):
        action = np.random.randint(env.num_actions) if step < 1000 else agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.store_transition(obs, action, reward, next_obs, done)
        ep_return += reward
        obs = next_obs
        if done:
            episode_returns.append(ep_return)
            ep_return = 0.0
            obs = env.reset()
        if step >= 1000:
            agent.train_step()
        if step % 2000 == 0 and episode_returns:
            print(f"[{run_name}] step={step} episodes={len(episode_returns)} "
                  f"mean_return(last10)={np.mean(episode_returns[-10:]):.2f} ({time.time()-t0:.0f}s elapsed)")
        if step % 10_000 == 0 and step > start_step:
            torch.save({"agent": agent.state_dict(), "step": step, "episode_returns": episode_returns}, ckpt_path)

    torch.save({"agent": agent.state_dict(), "step": num_steps, "episode_returns": episode_returns}, ckpt_path)
    print(f"[{run_name}] DONE ({time.time()-t0:.0f}s)")
    return {"episode_returns": episode_returns}


def load_existing(run_name):
    ckpt_path = os.path.join(CKPT_DIR, f"{run_name}.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {"episode_returns": ckpt.get("episode_returns", []), "rank_history": ckpt.get("rank_history", [])}


if __name__ == "__main__":
    results = {}
    overall_t0 = time.time()

    # results from earlier sessions, already on disk under their original filenames
    results["baseline_seed0"] = load_existing("baseline_breakout")
    results["spr_seed0"] = load_existing("spr_breakout")
    results["baseline_seed1"] = load_existing("baseline_breakout_seed1")
    results["spr_seed1"] = load_existing("spr_breakout_seed1")
    results["baseline_seed2"] = load_existing("baseline_breakout_seed2")
    results["spr_seed2"] = load_existing("spr_breakout_seed2")
    results["curl_seed0"] = load_existing("curl_breakout_seed0")
    results["spr_no_momentum_seed0"] = load_existing("spr_no_momentum_breakout_seed0")
    results["spr_no_augmentation_seed0"] = load_existing("spr_no_augmentation_breakout_seed0")
    results["atari_baseline"] = load_existing("atari_baseline_breakout")

    print("=" * 60, "\nSTAGE 5: IQRL at seeds 0, 1, 2\n", "=" * 60)
    for seed in (0, 1, 2):
        results[f"iqrl_seed{seed}"] = train_minatar_run(f"iqrl_{GAME}_seed{seed}", seed=seed, aux_kind="iqrl")

    print("=" * 60, "\nSTAGE 6: CURL at seeds 1, 2 (already have seed 0)\n", "=" * 60)
    for seed in (1, 2):
        results[f"curl_seed{seed}"] = train_minatar_run(f"curl_{GAME}_seed{seed}", seed=seed, aux_kind="curl")

    print("=" * 60, "\nSTAGE 7: SPR ablations at seed 1 (already have seed 0)\n", "=" * 60)
    results["spr_no_momentum_seed1"] = train_minatar_run(
        f"spr_no_momentum_{GAME}_seed1", seed=1, aux_kind="spr", spr_momentum_tau=0.0, spr_aug_pad=2)
    results["spr_no_augmentation_seed1"] = train_minatar_run(
        f"spr_no_augmentation_{GAME}_seed1", seed=1, aux_kind="spr", spr_momentum_tau=0.99, spr_aug_pad=0)

    summary_path = os.path.join(REPO_ROOT, "notebooks", "_overnight_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f)

    print(f"\nTOTAL WALL TIME: {(time.time()-overall_t0)/60:.1f} min")
    print("PHASE 2 COMPLETION PIPELINE COMPLETE")
