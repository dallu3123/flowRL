import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))

class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        return batch

    def sample_sequence(self, batch_size, sequence_length, discount):
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)
        
        data = {k: v[idxs] for k, v in self.items()}

        rewards = np.zeros(data['rewards'].shape + (sequence_length,), dtype=float)
        masks = np.ones(data['masks'].shape + (sequence_length,), dtype=float)
        valid = np.ones(data['masks'].shape + (sequence_length,), dtype=float)
        observations = np.zeros(data['observations'].shape[:-1] + (sequence_length, data['observations'].shape[-1]), dtype=float)
        next_observations = np.zeros(data['observations'].shape[:-1] + (sequence_length, data['observations'].shape[-1]), dtype=float)
        actions = np.zeros(data['actions'].shape[:-1] + (sequence_length, data['actions'].shape[-1]), dtype=float)
        terminals = np.zeros(data['terminals'].shape + (sequence_length,), dtype=float)

        for i in range(sequence_length):
            cur_idxs = idxs + i

            if i == 0:
                rewards[..., 0] = self['rewards'][cur_idxs]
                masks[..., 0] = self["masks"][cur_idxs]
                terminals[..., 0] = self["terminals"][cur_idxs]
            else:
                valid[..., i] = (1.0 - terminals[..., i - 1])
                rewards[..., i] = rewards[..., i - 1] + (self['rewards'][cur_idxs] * (discount ** i) * valid[..., i])
                masks[..., i] = np.minimum(masks[..., i-1], self["masks"][cur_idxs]) * valid[..., i] + masks[..., i-1] * (1. - valid[..., i])
                terminals[..., i] = np.maximum(terminals[..., i-1], self["terminals"][cur_idxs])
            
            actions[..., i, :] = self['actions'][cur_idxs]
            next_observations[..., i, :] = self['next_observations'][cur_idxs] * valid[..., i:i+1] + next_observations[..., i-1, :] * (1. - valid[..., i:i+1])
            observations[..., i, :] = self['observations'][cur_idxs]
            
        return dict(
            observations=data['observations'].copy(),
            actions=actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_observations,
        )

    def sample_traj(self, batch_size, h):
        """Sample time-major trajectory windows of h + 1 steps, as RQL expects.

        Ported from https://github.com/aoberai/rql/blob/main/utils/datasets.py (Dataset.sample_traj).
        Episodes are sampled uniformly (among those longer than one step), then a start step uniformly
        within the episode; indices past the episode end are clipped to its last step.
        Returns arrays of shape (h + 1, batch_size, ...).
        """
        idxs = np.random.choice(np.flatnonzero(self.terminal_locs - self.initial_locs), batch_size, replace=batch_size >= len(self.initial_locs))
        bt_initial_locs = self.initial_locs[idxs]
        bt_terminal_locs = self.terminal_locs[idxs]

        start_locs = np.random.randint(bt_initial_locs, bt_terminal_locs)

        keys = ['observations', 'actions', 'rewards', 'terminals', 'masks']
        steps = {k: [] for k in keys}
        for i in range(h + 1):
            cur_idxs = np.minimum(i + start_locs, bt_terminal_locs)
            for k in keys:
                steps[k].append(self[k][cur_idxs][None])

        return {k: np.concatenate(v, 0) for k, v in steps.items()}

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        return result

class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """
        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, pointer=0, size=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = size
        self.pointer = pointer

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        if transition['terminals'] > 0:
            # Keep episode boundaries current for sample_traj.
            self.terminal_locs = np.append(self.terminal_locs, self.pointer)
            self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0
