import abc
from unityagents import UnityEnvironment


class EnvInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def reset(self):
        """Reset environment and return starting state."""

    @abc.abstractmethod
    def step(self, action):
        """Do next step and return (state, reward, done info."""

    @abc.abstractmethod
    def close(self):
        """Close environment."""


class BananaMazeEnv(EnvInterface):
    def __init__(self, env_binary='../bin/unity_banana_maze/Banana.x86_64', train_mode=True):
        self.env = UnityEnvironment(file_name=env_binary)

        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.train_mode = train_mode
        self.info = self.env.reset(train_mode=self.train_mode)[self.brain_name]

    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        state = env_info.vector_observations[0]
        return state

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return (next_state, reward, done)

    def close(self):
        self.env.close()

    @property
    def state_size(self):
        state = self.info.vector_observations[0]
        return len(state)

    @property
    def action_size(self):
        action_size = self.brain.vector_action_space_size
        return action_size
