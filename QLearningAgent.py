import random
from collections import defaultdict

class DQLearningAgent:
    def __init__(self, epsilon=1, alpha =0.5, gamma=0.85):
        self.q1= defaultdict(float)
        self.q2 = defaultdict(float)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def get_q(self, state, action):
        return self.q1[(state,action)] + self.q2[(state,action)]
    
    def choose_action(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        else:
            q_values = [(a,self.get_q(state, a) )for a in available_actions]
            max_q = max(q_values, key=lambda x: x[1])[1]
            best_actions = [a for a,q in q_values if q == max_q]
            return random.choice(best_actions)
        
    def learn(self, state, action, reward, next_state, done, next_available_actions):
        if random.random() < 0.5:
            #update q1
            next_action = max(next_available_actions, key=lambda a: self.q1[(next_state, a)], default=None)
            next_q = self.q2[(next_state, next_action)] if next_action else 0
            td_target = reward if done else reward + self.gamma * next_q
            self.q1[(state, action)] += self.alpha * (td_target - self.q1[(state, action)])
        else:
            #update q2
            next_action = max(next_available_actions, key=lambda a: self.q2[(next_state, a)], default=None)
            next_q= self.q1[(next_state, next_action)] if next_action else 0
            td_target = reward if done else reward + self.gamma * next_q
            self.q2[(state, action)] += self.alpha * (td_target - self.q2[(state, action)])
