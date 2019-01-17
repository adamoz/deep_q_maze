import abc
import numpy as np
from qrl_navigation.model import QNetwork, DuelingQNetwork
from qrl_navigation.replay_buffer import ReplayBuffer, ReplayWeightedBuffer
import random
import torch
import torch.nn.functional as F
import torch.optim as optim


class AgentInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step(self, state, action, reward, next_state, done):
        """Save state to replay buffer and train if needed."""

    @abc.abstractmethod
    def act(self, state, eps=None, tau=None):
        """Return actions for given state as per current policy.

        Params:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
            tau (float): For soft update of target network parameters
        """


class Agent(AgentInterface):
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, fc_units=[64, 64], buffer_size=int(1e5), update_rate=4,
                 batch_size=64, tau=1e-3, eps=0., gamma=0.99, lr=5e-4, seed=0, device='cpu', dueling=True):
        """Initialize an Agent object.

        Params:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            fc_units (list): List of unit counts in each layer of q-netwokr
            buffer_size (int): Replay buffer size
            batch_size (int): Size of sampled batches from replay buffer
            lr (float): Learning rate
            gamma (float): Reward discount
            tau (float): For soft update of target network parameters
            eps (float): For epsilon-greedy action selection
            update_rate (int): Every update_rate step network params are updated

        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.update_rate = update_rate
        self.eps = eps
        self.state_size = state_size
        self.action_size = action_size
        self.fc_units = fc_units
        self.dueling = dueling

        # Q-Network
        if dueling:
            self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed, fc_units).to(device)
            self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed, fc_units).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed, fc_units).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed, fc_units).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, seed=seed)
        self.t_step = 0

    def __repr__(self):
        return f'Agent(state_size={self.state_size}, action_size={self.action_size}, fc_units={self.fc_units}, device="{self.device}", dueling={self.dueling})'

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn under update_rate.
        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0 and self.memory.is_ready_to_sample():
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, eps=None, tau=None):
        """Return actions for given state as per current policy.

        Params:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
            tau (float): For soft update of target network parameters
        """

        if eps is None:
            eps = self.eps

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma=None, tau=None):
        """Update value parameters using given batch of experience tuples.

        Params:
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            tau (float): For soft update of target network parameters
        """

        if gamma is None:
            gamma = self.gamma
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau)

    def soft_update(self, local_model, target_model, tau=None):
        """Soft update model parameters.

        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): For soft update of target network parameters
        """
        if tau is None:
            tau = self.tau

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, file_name):
        torch.save(self.qnetwork_local.state_dict(), file_name)

    def load(self, file_name):
        self.qnetwork_local.load_state_dict(torch.load(file_name))
        self.qnetwork_target.load_state_dict(torch.load(file_name))


class WeightedAgent(AgentInterface):
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, fc_units=[64, 64], buffer_size=int(1e5), update_rate=4,
                 batch_size=64, tau=1e-3, eps=0., gamma=0.99, lr=5e-4, seed=0, device='cpu', dueling=True):
        """Initialize an Agent object using weighted replay buffer.

        Params:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            fc_units (list): List of unit counts in each layer of q-netwokr
            buffer_size (int): Replay buffer size
            batch_size (int): Size of sampled batches from replay buffer
            lr (float): Learning rate
            gamma (float): Reward discount
            tau (float): For soft update of target network parameters
            eps (float): For epsilon-greedy action selection
            update_rate (int): Every update_rate step network params are updated

        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.update_rate = update_rate
        self.eps = eps
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.beta = 0.1

        # Q-Network
        if dueling:
            self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed, fc_units).to(device)
            self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed, fc_units).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed, fc_units).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed, fc_units).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.memory = ReplayWeightedBuffer(buffer_size=buffer_size, batch_size=batch_size, seed=seed)
        self.t_step = 0

    def __repr__(self):
        return f'WeightedAgent(state_size={self.state_size}, action_size={self.action_size}, fc_units={self.fc_units}, device="{self.device}", dueling={self.dueling})'

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        err = self._get_abs_err(state, action, reward, next_state, done)
        self.memory.add(state, action, reward, next_state, done, err)

        # Learn under update_rate.
        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0 and self.memory.is_ready_to_sample():
            experiences = self.memory.sample()
            self.learn(experiences)

    def _get_abs_err(self, state, action, reward, next_state, done):
        self.qnetwork_local.eval()
        self.qnetwork_target.eval()

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            next_state = torch.from_numpy(next_state).float().to(self.device).unsqueeze(0)
            action = torch.from_numpy(np.array([action])).long().to(self.device).unsqueeze(0)
            reward = torch.from_numpy(np.array([reward])).float().to(self.device).unsqueeze(0)
            done = torch.from_numpy(np.array([done])).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            Q_target_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)
            Q_target = reward + (self.gamma * Q_target_next * (1 - done))
            Q_expected = self.qnetwork_local(state).detach().gather(1, action)

            err = np.abs((Q_target - Q_expected).squeeze().numpy())

        self.qnetwork_local.train()
        self.qnetwork_target.train()
        return err

    def act(self, state, eps=None, tau=None):
        """Return actions for given state as per current policy.

        Params:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
            tau (float): For soft update of target network parameters
        """

        if eps is None:
            eps = self.eps

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma=None, tau=None):
        """Update value parameters using given batch of experience tuples.

        Params:
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            tau (float): For soft update of target network parameters
        """

        if gamma is None:
            gamma = self.gamma
        states, actions, rewards, next_states, dones, weights, idxs = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Weight with buffersize
        scaling_coef = 1 / (self.buffer_size * weights)
        scaling_coef = torch.pow(scaling_coef, self.beta)
        self.beta = min(1.005 * self.beta, 1)

        # Compute loss
        loss = torch.pow(Q_expected - Q_targets, 2)
        loss = scaling_coef * loss
        loss = torch.mean(loss)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target buffer weihts
        errs = self._get_abs_err(states, actions, rewards, next_states, dones)
        self.memory.update_errs(errs, idxs)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau)

    def soft_update(self, local_model, target_model, tau=None):
        """Soft update model parameters.

        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): For soft update of target network parameters
        """
        if tau is None:
            tau = self.tau

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, file_name):
        torch.save(self.qnetwork_local.state_dict(), file_name)

    def load(self, file_name):
        self.qnetwork_local.load_state_dict(torch.load(file_name))
        self.qnetwork_target.load_state_dict(torch.load(file_name))
