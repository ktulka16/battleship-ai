import numpy as np
import random
import pickle
from battleship_ai import QLearningAgent

class BattleshipEnv:
    def __init__(self, size=10):
        self.size = size
        self.ship_lengths = [5, 4, 3, 3, 2]
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size))
        self.ship_positions = np.zeros((self.size, self.size))
        self.hits = 0
        self.moves = 0
        for length in self.ship_lengths:
            self._place_ship(length)
        return self.board.copy()

    def _place_ship(self, length):
        placed = False
        while not placed:
            r, c = random.randint(0, 9), random.randint(0, 9)
            direction = random.choice(['H', 'V'])
            if direction == 'H' and c + length <= 10:
                if np.sum(self.ship_positions[r, c:c+length]) == 0:
                    self.ship_positions[r, c:c+length] = 1
                    placed = True
            elif direction == 'V' and r + length <= 10:
                if np.sum(self.ship_positions[r:r+length, c]) == 0:
                    self.ship_positions[r:r+length, c] = 1
                    placed = True

    def step(self, action):
        r, c = divmod(action, self.size)
        self.moves += 1
        done = False
        
        if self.board[r, c] != 0: 
            reward = -5.0 # Penalty for shooting same spot
        elif self.ship_positions[r, c] == 1: 
            self.board[r, c] = 1
            self.hits += 1
            reward = 5.0
            if self.hits == sum(self.ship_lengths):
                reward = 50.0
                done = True
        else: 
            self.board[r, c] = -1
            reward = -0.5

        if self.moves >= 100: done = True
        return self.board.copy(), reward, done

if __name__ == "__main__":
    env = BattleshipEnv()
    agent = QLearningAgent()
    episodes = 10000 

    print("Training started...")
    for e in range(1, episodes + 1):
        state = env.reset()
        for _ in range(100):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            if done: break
        
        if e % 1000 == 0:
            print(f"Episode: {e}, Epsilon: {agent.epsilon:.2f}")

    with open('model.pkl', 'wb') as f:
        pickle.dump(agent.q_table, f)
    print("Training complete. Table saved as 'model.pkl'.")
