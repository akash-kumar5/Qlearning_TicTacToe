import random

class RandomOpponent:
    def __init__(self, player=-1):
        self.player = player

    def choose_action(self, state, available_actions):
        return random.choice(available_actions)