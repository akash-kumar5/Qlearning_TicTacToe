import tkinter as tk
from tkinter import messagebox
import numpy as np
import os
import random
import pickle
from PerfectOpponent import PerfectOpponent  # Must implement: choose_action(board, player)

# ---------------- Tic Tac Toe Environment ----------------
class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        return tuple(self.board.flatten())

    def available_actions(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def step(self, action, player):
        i, j = action
        if self.board[i, j] != 0 or self.done:
            return self.get_state(), -10, True
        self.board[i, j] = player
        if self.check_winner(player):
            self.done = True
            self.winner = player
            return self.get_state(), 1 if player == 1 else -1, True
        elif len(self.available_actions()) == 0:
            self.done = True
            self.winner = 0
            return self.get_state(), 0.5, True
        else:
            return self.get_state(), 0, False

    def check_winner(self, player):
        for i in range(3):
            if np.all(self.board[i, :] == player): return True
            if np.all(self.board[:, i] == player): return True
        if np.all(np.diag(self.board) == player): return True
        if np.all(np.diag(np.fliplr(self.board)) == player): return True
        return False

# ---------------- Load Q-Table Agent ----------------
class PretrainedAgent:
    def __init__(self, q_table):
        self.q_table = q_table
        self.epsilon = 0  # no exploration during play

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_actions):
        q_values = [self.get_q(state, a) for a in available_actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
        return best_actions[0] if best_actions else random.choice(available_actions)

# ---------------- GUI ----------------
class TicTacToeGUI:
    def __init__(self, agent, use_perfect=False):
        self.agent = agent
        self.use_perfect = use_perfect
        self.env = TicTacToe()
        self.state = self.env.reset()

        self.window = tk.Tk()
        self.window.title("Tic Tac Toe - You (O) vs Agent (X)")
        self.buttons = [[None for _ in range(3)] for _ in range(3)]

        self.create_grid()
        self.window.after(100, self.agent_move)
        self.window.mainloop()

    def create_grid(self):
        for i in range(3):
            for j in range(3):
                btn = tk.Button(self.window, text="", font=('Arial', 24), width=4, height=2,
                                command=lambda i=i, j=j: self.human_move(i, j))
                btn.grid(row=i, column=j)
                self.buttons[i][j] = btn

    def human_move(self, i, j):
        if self.env.done or self.env.board[i][j] != 0:
            return
        self.update_board(i, j, -1)
        self.state, _, done = self.env.step((i, j), -1)
        self.check_game_over()
        if not done:
            self.window.after(30, self.agent_move)

    def agent_move(self):
        if self.use_perfect:
            flat_state = tuple(self.env.board.flatten())
            available = [i for i, val in enumerate(flat_state) if val == 0]
            action_index = self.agent.choose_action(flat_state, available)
            action = (action_index // 3, action_index % 3)
        else:
            action = self.agent.choose_action(self.state, self.env.available_actions())

        self.state, _, _ = self.env.step(action, 1)
        self.update_board(*action, 1)
        self.check_game_over()

    def update_board(self, i, j, player):
        symbol = 'O' if player == -1 else 'X'
        self.buttons[i][j]['text'] = symbol
        self.buttons[i][j]['state'] = 'disabled'

    def check_game_over(self):
        if self.env.done:
            for i in range(3):
                for j in range(3):
                    self.buttons[i][j]['state'] = 'disabled'

            if self.env.winner == 1:
                msg = "Agent wins! Play again?"
            elif self.env.winner == -1:
                msg = "You win! Play again?"
            else:
                msg = "It's a draw! Play again?"

            if messagebox.askyesno("Game Over", msg):
                self.restart_game()

    def restart_game(self):
        self.env = TicTacToe()
        self.state = self.env.reset()
        for i in range(3):
            for j in range(3):
                self.buttons[i][j]['text'] = ""
                self.buttons[i][j]['state'] = "normal"
        self.window.after(200, self.agent_move)

# ---------------- Load Pretrained Q Agent ----------------
def load_q_agent(filename='agent.pkl'):
    if not os.path.exists(filename):
        raise FileNotFoundError("Q-learning agent file not found.")
    with open(filename, 'rb') as f:
        q_table = pickle.load(f)
    return PretrainedAgent(q_table)

# ---------------- Main ----------------
if __name__ == "__main__":
    USE_MINIMAX = False  # Change to False to use pretrained Q-learning agent

    if USE_MINIMAX:
        agent = PerfectOpponent()
        TicTacToeGUI(agent, use_perfect=True)
    else:
        agent = load_q_agent()
        TicTacToeGUI(agent, use_perfect=False)
