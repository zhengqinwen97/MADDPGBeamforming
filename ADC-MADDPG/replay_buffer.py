import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            obs_t, action, reward, obs_tp1, done = self._storage[i]
            obses_t.append(obs_t)
            actions.append(action)
            rewards.append(reward)
            obses_tp1.append(obs_tp1)
            dones.append(done)
        return obses_t, np.array(actions), np.array(rewards), obses_tp1, np.array(dones)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
    