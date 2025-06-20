import math
import numpy as np
from functools import lru_cache

class PerfectOpponent:
    def __init__(self, player=-1):
        self.player = player  # -1 or 1
        self.original_player = player

    def choose_action(self, state, available_actions):
        # if self.original_player == -1:
        #     state = self._invert_state(state)
        best_score = -math.inf
        best_move = None
        for action in available_actions:
            next_state = self._make_move(state, action, self.player)
            score = self._minimax_cached(next_state, False, self.player)
            if score > best_score:
                best_score = score
                best_move = action
        return best_move

    def _invert_state(self, state):
        """Convert canonical state back to original perspective"""
        return tuple(-x if x != 0 else 0 for x in state)
    

    @lru_cache(maxsize=None)
    def _minimax_cached(self, state, is_maximizing, current_player):
        state_array = np.array(state).reshape(3, 3)
        winner = self._check_winner(state_array)
        if winner is not None:
            return winner * self.player  # Return from perspective of our player

        available = self._get_available_actions(state_array)
        if not available:
            return 0  # draw

        if is_maximizing:
            value = -math.inf
            for action in available:
                next_state = self._make_move(state, action, current_player)
                value = max(value, self._minimax_cached(next_state, False, current_player))
            return value
        else:
            value = math.inf
            for action in available:
                next_state = self._make_move(state, action, -current_player)
                value = min(value, self._minimax_cached(next_state, True, current_player))
            return value

    def _make_move(self, state, action, player):
        """Returns a new state with the move made"""
        state = np.array(state).copy().reshape(3, 3)
        i, j = divmod(action, 3)
        if state[i, j] != 0:
            raise ValueError(f"Invalid move: position {action} is not empty")
        state[i, j] = player
        return tuple(state.flatten())

    def _get_available_actions(self, state):
        """Returns list of available actions (indices)"""
        return [i for i in range(9) if state.flatten()[i] == 0]

    def _check_winner(self, state):
        """Returns 1, -1 for winner, None otherwise"""
        lines = [
            (0,1,2), (3,4,5), (6,7,8),  # rows
            (0,3,6), (1,4,7), (2,5,8),  # columns
            (0,4,8), (2,4,6)            # diagonals
        ]
        state_flat = state.flatten()
        for a, b, c in lines:
            if state_flat[a] != 0 and state_flat[a] == state_flat[b] == state_flat[c]:
                return state_flat[a]
        return None