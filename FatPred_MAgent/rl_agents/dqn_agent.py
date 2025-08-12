# Simple placeholder DQN agent (real DQN requires neural network implementation)
import numpy as np



class DQNAgent:
    def __init__(self, actions):
        self.actions = actions
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
        action = max(self.q_table[state], key=self.q_table[state].get, default=np.random.choice(self.actions))
        self.last_action = action
        return action

    def update_q_table(self, state, action, reward, next_state):
        self.q_table.setdefault(state, {a: 0 for a in self.actions})
        self.q_table.setdefault(next_state, {a: 0 for a in self.actions})
        self.q_table[state][action] = reward  # No real DQN update, placeholder only
        self.last_reward = reward
