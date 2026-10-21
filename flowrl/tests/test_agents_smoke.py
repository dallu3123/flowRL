"""CPU smoke tests: every agent can be created, updated (offline and UTD batch updates) and queried for actions.

Uses a small synthetic dataset, so no OGBench download is needed. Networks are shrunk to keep the test fast.
"""
import importlib

import jax
import numpy as np
import pytest

from agents import agents
from utils.datasets import Dataset, ReplayBuffer

OBS_DIM, ACTION_DIM, HORIZON, BATCH_SIZE, UTD = 7, 3, 3, 8, 2


def make_dataset(num_episodes=6, episode_len=20, seed=0):
    rng = np.random.default_rng(seed)
    size = num_episodes * episode_len
    terminals = np.zeros(size, dtype=np.float32)
    terminals[episode_len - 1::episode_len] = 1.0
    return dict(
        observations=rng.normal(size=(size, OBS_DIM)).astype(np.float32),
        actions=rng.uniform(-1, 1, size=(size, ACTION_DIM)).astype(np.float32),
        rewards=rng.normal(size=size).astype(np.float32),
        terminals=terminals,
        masks=1.0 - terminals,
        next_observations=rng.normal(size=(size, OBS_DIM)).astype(np.float32),
    )


def make_config(name):
    config = importlib.import_module(f'agents.{name}').get_config()
    config['batch_size'] = BATCH_SIZE
    config['horizon_length'] = HORIZON
    if 'action_chunking' in config:
        config['action_chunking'] = True
    for key in ['actor_hidden_dims', 'value_hidden_dims']:
        if key in config:
            config[key] = (32, 32)
    for key in ['num_qs', 'ensemble_ct']:
        if key in config:
            config[key] = 2
    if 'h' in config:
        config['h'] = HORIZON
    return config


def sample_batch(ds, config):
    if config.get('batch_format', 'sequence') == 'traj':
        return ds.sample_traj(BATCH_SIZE, h=config['h'])
    return ds.sample_sequence(BATCH_SIZE, sequence_length=HORIZON, discount=config['discount'])


def sample_utd_batch(ds, config):
    batches = [sample_batch(ds, config) for _ in range(UTD)]
    return {k: np.stack([b[k] for b in batches]) for k in batches[0]}


@pytest.mark.parametrize('name', sorted(agents))
def test_agent_update_and_act(name):
    np.random.seed(0)
    config = make_config(name)
    ds = Dataset.create(**make_dataset())
    example = ds.sample(())
    agent = agents[name].create(0, example['observations'], example['actions'], config)

    update_kwargs = {'full_update': True} if name == 'rebrac' else {}

    agent, info = agent.update(sample_batch(ds, config), **update_kwargs)
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(info)), info

    replay = ReplayBuffer.create_from_initial_dataset(dict(ds), size=ds.size + 50)
    agent, info = agent.batch_update(sample_utd_batch(replay, config), **update_kwargs)
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(info)), info

    actions = np.asarray(agent.sample_actions(observations=example['observations'], rng=jax.random.PRNGKey(1)))
    assert actions.size == ACTION_DIM * HORIZON, actions.shape
    assert np.all(np.abs(actions) <= 1.0)


def test_replay_buffer_tracks_new_episodes_for_traj_sampling():
    ds = Dataset.create(**make_dataset(num_episodes=2, episode_len=5))
    replay = ReplayBuffer.create_from_initial_dataset(dict(ds), size=ds.size + 10)
    example = {k: v[0] for k, v in ds.items()}
    for step in range(4):
        replay.add_transition({**example, 'terminals': float(step == 3), 'masks': 1.0})

    np.testing.assert_array_equal(replay.terminal_locs, [4, 9, 13])
    np.testing.assert_array_equal(replay.initial_locs, [0, 5, 10])

    np.random.seed(0)
    batch = replay.sample_traj(64, h=2)
    assert batch['observations'].shape == (3, 64, OBS_DIM)
