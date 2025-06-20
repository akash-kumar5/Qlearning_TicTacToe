import pickle
import math
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np
from QLearningAgent import DQLearningAgent
from TicTacToe import TicTacToe
from PerfectOpponent import PerfectOpponent
from RandomOpponent import RandomOpponent

def canonical_form(state, player):
    board = np.array(state).reshape(3, 3)
    board = board * player  # normalize to always view from agent's POV

    forms = []
    for k in range(4):
        rotated = np.rot90(board, k)
        forms.append(tuple(rotated.flatten()))
        forms.append(tuple(np.fliplr(rotated).flatten()))

    return min(forms)

def load_agent(filename="agent.pkl"):
    agent = DQLearningAgent()
    with open(filename, "rb") as f:
        agent.q1, agent.q2 = pickle.load(f)
    return agent

def save_agent(agent, filename="agent.pkl"):
    with open(filename, "wb") as f:
        pickle.dump((agent.q1, agent.q2), f)

def get_epsilon(episode, min_epsilon=0.01, max_epsilon=1.0, decay_rate=0.0003):
    return max(min_epsilon, max_epsilon * math.exp(-decay_rate * episode))

def self_play_training(agent, episodes=10000):
    history = {'episodes': [], 'win_rate': [], 'loss_rate': [], 'draw_rate': [], 'epsilon': []}
    best_loss_rate = 100

    for ep in trange(episodes):
        agent.epsilon = get_epsilon(ep)
        player = 1
        env = TicTacToe()
        state = env.reset()
        original_state = state
        state = canonical_form(state, player)
        done = False
        opponent = PerfectOpponent(player=-1)

        while not done:
            available = env.available_actions()
            if player == 1:
                action = agent.choose_action(state, available)
            else:
                action = opponent.choose_action(original_state, available)

            next_state, reward, done = env.step(action, player)
            original_state = next_state
            next_state = canonical_form(next_state, player)

            if done:
                if env.winner == 1:
                    reward = 1
                elif env.winner == -1:
                    reward = -1
                else:
                    reward = 0
            else:
                reward = -0.01

            if player == 1:
                next_available = env.available_actions() if not done else []
                agent.learn(state, action, reward, next_state, done, next_available)

            state = next_state
            player *= -1

        if ep % 1000 == 0:
            eval_results = evaluate(agent)
            total = sum(eval_results.values())
            win_rate = eval_results['win'] / total * 100
            loss_rate = eval_results['loss'] / total * 100
            draw_rate = eval_results['draw'] / total * 100

            if loss_rate < best_loss_rate:
                best_loss_rate = loss_rate
                print(f"New best model saved -> Win: {win_rate:.2f}% | Loss: {loss_rate:.2f}% | Draw: {draw_rate:.2f}% at {ep}")
                save_agent(agent)
                agent = load_agent()

            history['episodes'].append(ep)
            history['win_rate'].append(win_rate)
            history['loss_rate'].append(loss_rate)
            history['draw_rate'].append(draw_rate)
            history['epsilon'].append(agent.epsilon)

    return agent, history

def evaluate(agent, opponent=PerfectOpponent(), games=1000):
    env = TicTacToe()
    results = {'win': 0, 'loss': 0, 'draw': 0}

    for _ in range(games):
        player = 1
        state= env.reset()
        state = canonical_form(state, player)
        done = False

        while not done:
            available = env.available_actions()
            if player == 1:
                action = agent.choose_action(state, available)
            else:
                action = opponent.choose_action(state, available)
            state, _, done = env.step(action, player)
            state = canonical_form(state, player)
            player *= -1

        if env.winner == 1:
            results['win'] += 1
        elif env.winner == -1:
            results['loss'] += 1
        else:
            results['draw'] += 1

    return results

def evaluate_opponent_vs(opponent1, opponent2, games=1000):
    results = {'win': 0, 'loss': 0, 'draw': 0}

    for _ in range(games):
        env = TicTacToe()
        state = env.reset()
        done = False
        player = 1

        while not done:
            available = env.available_actions()
            if player == 1:
                action = opponent1.choose_action(state, available)
            else:
                action = opponent2.choose_action(state, available)
            if state[action] != 0:
                raise Exception(f"Invalid move: {action} not available in {available}")
            state, _, done = env.step(action, player)
            player *= -1

        if env.winner == 1:
            results['win'] += 1
        elif env.winner == -1:
            results['loss'] += 1
        else:
            results['draw'] += 1

    return results

def plot_history(history):
    plt.plot(history['episodes'], history['win_rate'], label='Win %')
    plt.plot(history['episodes'], history['loss_rate'], label='Loss %')
    plt.plot(history['episodes'], history['draw_rate'], label='Draw %')
    plt.plot(history['episodes'], history['epsilon'], label='epsilon')
    plt.xlabel("Episodes")
    plt.ylabel("Rate (%)")
    plt.title("Agent Performance Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # agent = DQLearningAgent()
    # agent, history = self_play_training(agent)

    agent = PerfectOpponent(player=1)

    print("\nFinal Evaluation vs RandomOpponent:")
    print(evaluate_opponent_vs(agent, RandomOpponent(player=-1)))

    print("\nFinal Evaluation vs PerfectOpponent:")
    print(evaluate_opponent_vs(agent, PerfectOpponent(player=-1)))

    # plot_history(history)