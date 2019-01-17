from collections import deque
import numpy as np
import os
import pickle


def dqn_training(agent, env, save_path, n_episodes=2000, n_max_steps_per_episode=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, logger=None):
    """Deep Q-Learning.

    Params:
        n_episodes (int): maximum number of training episodes
        n_max_steps_per_episode (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    printer = print
    if logger is not None:
        printer = logger.info

    meta = {
        'scores': [],
        'eps': [],
        'meta_path': os.path.join(save_path, 'meta.pickle'),
        'model_path': os.path.join(save_path, 'model.pt'),
        'agent': repr(agent)
    }

    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    best_mean_score = 0

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0

        for step in range(n_max_steps_per_episode):
            action = agent.act(state, eps)
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        eps = max(eps_end, eps_decay * eps)
        printer(f'Current eps is {eps}')

        scores_window.append(score)
        meta['scores'].append(score)
        meta['eps'].append(eps)
        mean_score = np.mean(scores_window)

        printer(f'\rEpisode {i_episode}\tAverage Score: {mean_score}')
        if mean_score >= best_mean_score:
            best_mean_score = mean_score
            agent.save(meta['model_path'])

    with open(meta['meta_path'], 'wb') as wb:
        pickle.dump(meta, wb)
    printer(f'\rWe have achieved {best_mean_score} mean score (13+ is good result)')
    return meta
