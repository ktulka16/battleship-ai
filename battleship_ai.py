import numpy as np
import random

class QLearningAgent:
    def __init__(self, size=10):
        self.size = size
        self.q_table = {}
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.lr = 0.1
        self.gamma = 0.9

    def _get_state_key(self, board):
        return str(board.flatten())

    def act(self, board):
        state_key = self._get_state_key(board)
        flattened_board = board.flatten()
        
        # Identify "legal" moves (where board == 0)
        legal_moves = [i for i, val in enumerate(flattened_board) if val == 0]
        
        # If no legal moves left (shouldn't happen), pick a random one
        if not legal_moves:
            return random.randint(0, 99)

        # Initialize state in table if new
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.size * self.size)

        # EXPLORE: Pick a random move ONLY from legal options
        if np.random.rand() < self.epsilon:
            return random.choice(legal_moves)
        
        # EXPLOIT: Pick the best move from legal options
        # We temporarily set illegal moves to a very low value so they aren't chosen
        q_values = self.q_table[state_key].copy()
        mask = np.full(100, -np.inf)
        mask[legal_moves] = 0
        return np.argmax(q_values + mask)

    def learn(self, state, action, reward, next_state, done):
        s_key = self._get_state_key(state)
        ns_key = self._get_state_key(next_state)

        if s_key not in self.q_table: self.q_table[s_key] = np.zeros(100)
        if ns_key not in self.q_table: self.q_table[ns_key] = np.zeros(100)

        target = reward + (1 - done) * self.gamma * np.max(self.q_table[ns_key])
        self.q_table[s_key][action] += self.lr * (target - self.q_table[s_key][action])

        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
