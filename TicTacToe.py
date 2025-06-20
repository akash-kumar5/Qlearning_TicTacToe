import numpy as np

class TicTacToe:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = np.zeros((3,3), dtype=int)
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        return tuple(self.board.flatten())
    
    def available_actions(self):
            return [i * 3 + j for i in range(3) for j in range(3) if self.board[i, j] == 0]
    
    def step(self,action , player):
        i,j = divmod(action, 3)
        if self.board[i,j] != 0:
            return self.get_state(), -10, False
        
        self.board[i,j] = player
        if self.check_winner(player):
            self.done= True
            self.winner = player
            return self.get_state(), 10 if player == 1 else -3 , True
        elif len(self.available_actions()) == 0:
            self.done = True
            self.winner = 0
            return self.get_state(), 5, True
        else:
            return self.get_state(), -1 , False
        
    def check_winner(self,player):
        for i in range(3):
            if np.all(self.board[i, :] == player): return True
            if np.all(self.board[:,i]==player): return True
        if np.all(np.diag(self.board) == player): return True
        if np.all(np.diag(np.fliplr(self.board)) == player): return True
        return False
    
    def render(self):
        symbols = {1: 'X', -1:'O', 0:' '}
        for row in self.board:
            print(' | '.join(symbols[val] for val in row))
            print('- '*5)


env = TicTacToe()
env.render()
print("Available moves:", env.available_actions())

_, reward, done = env.step(0, 1)
env.render()
print("Reward:", reward, "Done:", done)
