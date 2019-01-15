from collections import namedtuple, deque
import numpy as np
from numpy.random import choice
import random
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size=int(1e5), batch_size=64, seed=0, device='cpu'):
        """Initialize a ReplayBuffer object.

        Params:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (str): device where tensors are proecssed
        """

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def is_ready_to_sample(self):
        return len(self) > self.batch_size

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class ReplayWeightedBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size=int(1e5), batch_size=64, seed=0, device='cpu'):
        """Initialize a ReplayWeightedBuffer object with weighted priorities.

        Params:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (str): device where tensors are proecssed
        """

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device
        self.seed = random.seed(seed)
        self.buffer_size = buffer_size
        self.errs = list()

    def add(self, state, action, reward, next_state, done, err):
        """Add a new experience to memory with obtained error."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.errs.append(np.abs(err) + 0.01)
        self.errs = self.errs[:self.buffer_size]

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        all_probs = np.array(self.errs) / sum(self.errs)
        idxs = choice(range(len(self.errs)), size=self.batch_size, p=all_probs)
        experiences = [self.memory[idx] for idx in idxs]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        probs = torch.from_numpy(np.vstack([all_probs[idx] for idx in idxs])).float().to(self.device)

        return (states, actions, rewards, next_states, dones, probs, idxs)

    def update_errs(self, values, indexes):
        self.errs = np.array(self.errs)
        self.errs[indexes] = values + 0.01
        self.errs = self.errs.tolist()

    def is_ready_to_sample(self):
        return len(self) > self.batch_size

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
