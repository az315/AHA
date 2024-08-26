import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
import os
from .callbacks import state_to_tuple, ACTIONS

ALPHA = 0.1  # Erhöht die Lernrate für schnelleres Lernen
GAMMA = 0.9  # Erhöht den Diskontfaktor, um langfristige Belohnungen stärker zu gewichten

def setup_training(self):
    
    
    self.transitions = []
    self.epsilon_start = 1.0  # Anpassung der Start-Exploration
    self.epsilon_end = 0.1  # Anpassung der End-Exploration
    self.epsilon_decay = 0.995

    self.episode = 0
    self.epsilon = self.epsilon_start

    self.episodes = 10000  # Anzahl der Episoden festlegen
    self.max_steps = 100  # Maximale Schritte pro Episode

    # Setup logger
    logging.basicConfig(filename='training.log', level=logging.INFO)
    self.logger = logging.getLogger()

def reward_from_events(self, events):
    """
    Calculate reward from events.
    """
    game_rewards = {
        'COIN_COLLECTED': 30,
        'KILLED_OPPONENT': 10,
        'MOVED_TOWARDS_COIN': 6,
        'INVALID_ACTION': -1,
        'GOT_KILLED': -10,
        'KILLED_SELF': -10,
        'WAITED_TOO_LONG': 0
    }
    reward = sum([game_rewards.get(event, 0) for event in events])
    return reward

def plot_rewards(rewards):
    plt.plot(rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

rewards = []

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events):
    """
    Called once per step to allow intermediate rewards.
    """
    if old_game_state is None or new_game_state is None:
        return
    
    old_state = state_to_tuple(old_game_state)
    new_state = state_to_tuple(new_game_state)
    
    # Initialisiere Zustände in Q-Tabelle
    if old_state not in self.q_table:
        self.q_table[old_state] = np.zeros(len(ACTIONS))
    if new_state not in self.q_table:
        self.q_table[new_state] = np.zeros(len(ACTIONS))
    
    action_index = ACTIONS.index(self_action)
    reward = reward_from_events(self, events)
    
    old_q = self.q_table[old_state][action_index]
    future_q = np.max(self.q_table[new_state])
    
    # Q-Wert-Update
    new_q = old_q + ALPHA * (reward + GAMMA * future_q - old_q)
    self.q_table[old_state][action_index] = new_q

def end_of_round(self, last_game_state: dict, last_action: str, events):
    """
    Called at the end of each game or when the agent dies.
    """
    game_events_occurred(self, last_game_state, last_action, None, events)
    
    # Update Epsilon
    self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    total_reward = reward_from_events(self, events)
    self.logger.info(f"Episode finished with epsilon: {self.epsilon}, reward: {total_reward}")
    rewards.append(total_reward)

    self.episode += 1

    if len(rewards) % 500 == 0:
        plot_rewards(rewards)

    # Speichere die Q-Tabelle nach jedem 100. Episode
    if self.episode % 100 == 0:
        with open("q_table.pkl", "wb") as file:
            pickle.dump(self.q_table, file)

def evaluate_agent(self):
    """
    Run a series of episodes without exploration to evaluate the agent's current performance.
    """
    original_epsilon = self.epsilon
    self.epsilon = 0  # Turn off exploration for evaluation

    # Run evaluation
    for _ in range(10):  # Run 10 episodes for evaluation
        # Implement the evaluation loop
        pass  # Add your evaluation logic here

    self.epsilon = original_epsilon  # Restore exploration
