import numpy as np


class SarsaAgent:
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
        next_action = self.choose_action(next_state)
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.last_reward = reward
