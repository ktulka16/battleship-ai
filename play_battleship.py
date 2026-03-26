import numpy as np
import pickle
import random
from battleship_ai import QLearningAgent

class GameInterface:
    def __init__(self, model_path='model.pkl'):
        self.size = 10
        self.ship_lengths = [5, 4, 3, 3, 2]
        self.board = np.zeros((self.size, self.size)) # AI's board (what you see)
        self.ship_positions = np.zeros((self.size, self.size)) # Secret ship locations
        self.hits = 0
        self.total_cells = sum(self.ship_lengths)
        
        # Load the trained AI
        try:
            with open(model_path, 'rb') as f:
                self.q_table = pickle.load(f)
            print("--- AI Brain Loaded Successfully ---")
        except FileNotFoundError:
            self.q_table = {}
            print("--- No Model Found: AI will play randomly ---")

    def _place_ships(self):
        for length in self.ship_lengths:
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

    def display(self):
        print("\n   0 1 2 3 4 5 6 7 8 9")
        for r in range(self.size):
            row_str = f"{r} "
            for c in range(self.size):
                val = self.board[r, c]
                if val == 0: row_str += " ~" # Unknown
                elif val == 1: row_str += " H" # Hit
                else: row_str += " M" # Miss
            print(row_str)

    def ai_turn(self):
        # Logic to pick the best move from the loaded Q-Table
        state_key = str(self.board.flatten())
        legal_moves = [i for i, val in enumerate(self.board.flatten()) if val == 0]
        
        if state_key in self.q_table:
            q_values = self.q_table[state_key].copy()
            mask = np.full(100, -np.inf)
            mask[legal_moves] = 0
            action = np.argmax(q_values + mask)
        else:
            action = random.choice(legal_moves)
            
        return action

    def play(self):
        self._place_ships()
        print("Welcome to Battleship!")
        
        while self.hits < self.total_cells:
            self.display()
            try:
                move = input("\nEnter coordinates (row col) or 'q' to quit: ")
                if move.lower() == 'q': break
                
                r, c = map(int, move.split())
                if self.board[r, c] != 0:
                    print("You already shot there! Try again.")
                    continue
                
                if self.ship_positions[r, c] == 1:
                    print(">>> HIT! <<<")
                    self.board[r, c] = 1
                    self.hits += 1
                else:
                    print(">>> MISS <<<")
                    self.board[r, c] = -1
                    
            except (ValueError, IndexError):
                print("Invalid input. Please enter numbers from 0-9 (e.g., '5 5').")

        if self.hits == self.total_cells:
            self.display()
            print("\nVICTORY! You sank the AI's fleet.")

if __name__ == "__main__":
    game = GameInterface()
    game.play()
