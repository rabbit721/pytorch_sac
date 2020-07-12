import numpy as np
import torch


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, latent_dim, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.int32

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.latent_vecs = np.empty((capacity, latent_dim), dtype=np.float32)
        self.actions = np.empty((capacity, 1), dtype=np.int32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, latent_vec, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.latent_vecs[self.idx], latent_vec)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def _tensor_from_idxs(self, idxs):
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        latent_vecs = torch.as_tensor(self.latent_vecs[idxs], device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return obses, latent_vecs, actions, rewards, next_obses, not_dones, not_dones_no_max

    def get_latest_batch(self, batch_size):
        if self.idx <= batch_size:
            if self.full:
                idxs = np.array([i for i in range(self.idx)] \
                                 + [self.capacity - i - 1 for i in range(batch_size - self.idx)])
            else:
                idxs = np.array([i for i in range(self.idx)])
        else:
            idxs = np.arange(self.idx - batch_size, self.idx)
        return self._tensor_from_idxs(idxs)

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        return self._tensor_from_idxs(idxs)
