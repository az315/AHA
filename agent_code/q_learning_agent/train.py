import numpy as np
import pickle
import os
from .callbacks import state_to_tuple, ACTIONS

ALPHA = 0.1
GAMMA = 0.99

def setup_training(self):
    """
    This is called once when the training is started.
    Use this function to initialize training-specific settings or variables.
    """
    # Hier kannst du Variablen initialisieren, die nur während des Trainings benötigt werden
    self.transitions = []
    self.epsilon_start = 1.0
    self.epsilon_end = 0.1
    self.epsilon_decay = 0.995
    self.epsilon = self.epsilon_start
    self.episodes = 10000  # Anzahl der Episoden festlegen
    self.max_steps = 100  # Maximale Schritte pro Episode

def reward_from_events(self, events):
    """
    Calculate reward from events.
    """
    game_rewards = {
        'COIN_COLLECTED': 5,
        'KILLED_OPPONENT': 10,
        'MOVED_TOWARDS_COIN': 1,
        'INVALID_ACTION': -5,
        'GOT_KILLED': -10,
        'KILLED_SELF': -10
    }
    reward = sum([game_rewards.get(event, 0) for event in events])
    return reward

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
    # Rufe die Funktion `game_events_occurred` direkt auf, ohne `self` als Aufrufer
    game_events_occurred(self, last_game_state, last_action, None, events)
    
    # Update Epsilon
    self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    # Speichere die Q-Tabelle
    with open("q_table.pkl", "wb") as file:
        pickle.dump(self.q_table, file)


