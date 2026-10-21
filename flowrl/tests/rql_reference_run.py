"""Run one RQL step from a given codebase root and save the outputs, for comparing the port against upstream.

Usage: python rql_reference_run.py <codebase_root> <dataset.npz> <out.npz>

Both codebases use top-level `agents` / `utils` packages, so each root runs in its own process.
"""
import inspect
import os
import sys

root, dataset_path, out_path = sys.argv[1:4]
sys.path.insert(0, os.path.abspath(root))

import jax
import numpy as np

from agents.rql import RQLAgent, get_config
from utils.datasets import Dataset

H, BATCH_SIZE, SEED = 3, 16, 0

config = get_config()
config['h'] = H
config['batch_size'] = BATCH_SIZE
config['actor_hidden_dims'] = (32, 32)
config['value_hidden_dims'] = (32, 32)
config['ensemble_ct'] = 2
config['rho'] = 0.5
config['expectile'] = 0.7

ds = Dataset.create(**dict(np.load(dataset_path)))
np.random.seed(SEED)
if hasattr(ds, 'sample_sequence'):  # flowrl
    batch = ds.sample_traj(BATCH_SIZE, h=H)
else:  # upstream dispatches to sample_traj through dataset.config
    ds.config = config
    batch = ds.sample(BATCH_SIZE)

agent = RQLAgent.create(SEED, batch['observations'], batch['actions'], config)
loss, info = agent.total_loss(batch, agent.network.params, rng=jax.random.PRNGKey(7))
new_agent, update_info = agent.update(batch)

obs = batch['observations'][0, 0]
act_rng = jax.random.PRNGKey(3)
if 'rng' in inspect.signature(RQLAgent.sample_actions).parameters:
    actions = new_agent.sample_actions(observations=obs, rng=act_rng, temperature=0.0)
else:
    actions = new_agent.sample_actions(obs=obs, seed=act_rng, temperature=0.0)

out = {f'batch/{k}': np.asarray(v) for k, v in batch.items()}
out['loss'] = np.asarray(loss)
out.update({f'info/{k}': np.asarray(v) for k, v in info.items()})
out.update({f'update_info/{k}': np.asarray(v) for k, v in update_info.items()})
out['actions'] = np.asarray(actions)
for path, leaf in jax.tree_util.tree_flatten_with_path(new_agent.network.params)[0]:
    out['params/' + jax.tree_util.keystr(path)] = np.asarray(leaf)
np.savez(out_path, **out)
