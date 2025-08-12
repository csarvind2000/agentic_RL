import numpy as np


class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def get_state(self, data):
        missing_ratio = round(data.isnull().mean().mean(), 2)
        return (missing_ratio,)

    def choose_action(self, state):
        self.last_state = state
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}

        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)

        self.last_action = action
        return action

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.actions}

        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        self.q_table[state][action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.last_reward = reward
