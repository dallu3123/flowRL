"""Check that the RQL port matches upstream (github.com/aoberai/rql) on the same data.

Set RQL_UPSTREAM to a checkout of the upstream repo, e.g.
    git clone https://github.com/aoberai/rql /tmp/rql && git -C /tmp/rql checkout 229c956efb4494c2b9bb0bbddbd67b761c93f1cc
    RQL_UPSTREAM=/tmp/rql pytest tests/test_rql_port.py
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from test_agents_smoke import make_dataset

FLOWRL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(FLOWRL_ROOT, 'tests', 'rql_reference_run.py')
UPSTREAM = os.environ.get('RQL_UPSTREAM')


def run(root, dataset_path, out_path):
    env = {**os.environ, 'JAX_PLATFORMS': 'cpu'}
    # Run from a neutral directory so each process only sees its own root's packages.
    subprocess.run([sys.executable, RUNNER, root, dataset_path, out_path], check=True, env=env, cwd=os.path.dirname(out_path))
    return dict(np.load(out_path))


@pytest.mark.skipif(not UPSTREAM, reason='RQL_UPSTREAM is not set')
def test_rql_port_matches_upstream(tmp_path):
    dataset_path = str(tmp_path / 'dataset.npz')
    np.savez(dataset_path, **make_dataset(num_episodes=8, episode_len=15))

    ours = run(FLOWRL_ROOT, dataset_path, str(tmp_path / 'ours.npz'))
    theirs = run(UPSTREAM, dataset_path, str(tmp_path / 'theirs.npz'))

    assert ours.keys() == theirs.keys()
    for key in ours:
        np.testing.assert_allclose(ours[key], theirs[key], rtol=1e-6, atol=1e-6, err_msg=key)
