import numpy as np
import os
import pickle

from AHA.agent_code.q_learning_agent.a_star import find_nearest_coin_with_a_star

#ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

def setup(self):
    """
    Setup your agent. This method is called once when loading each agent.
    Use this to initialize variables, models, etc.
    """
    self.q_table = {}
    self.epsilon_start = 1.0
    self.epsilon_end = 0.1
    self.epsilon_decay = 0.995
    self.epsilon = self.epsilon_start

    # Lade vorhandene Q-Tabelle
    try:
        if os.path.isfile("q_table.pkl"):
            with open("q_table.pkl", "rb") as file:
                self.q_table = pickle.load(file)
    except (EOFError, pickle.UnpicklingError):
        self.logger.info("Q-table konnte nicht geladen werden, starte mit leerer Tabelle.")
        self.q_table = {}

def state_to_tuple(game_state: dict) -> tuple:
    """
    Simplified state representation.
    """
    if game_state is None:
        return None
    
    own_position = game_state['self'][3]
    field = game_state['field']
    coins = game_state['coins']
    bombs = game_state['bombs']
    
    # Berechne relative Position zur nächsten Münze
    coin_rel_pos = find_nearest_coin_with_a_star(own_position, coins, field)

    # Umgebungsinformationen sammeln
    surroundings = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # LEFT, RIGHT, UP, DOWN
    for dx, dy in directions:
        x, y = own_position[0] + dx, own_position[1] + dy
        if 0 <= x < field.shape[0] and 0 <= y < field.shape[1]:
            surroundings.append(field[x, y])
        else:
            surroundings.append(-1)  # Außerhalb des Feldes
    
    # Bombeninformationen
    bomb_info = []
    for bomb in bombs:
        bomb_pos, bomb_timer = bomb
        bomb_distance = np.linalg.norm(np.array(bomb_pos) - np.array(own_position))
        bomb_info.append((bomb_distance, bomb_timer))
    
    # Kann Bombe gelegt werden
    can_place_bomb = game_state['self'][2]
    
    return (
        own_position,
        coin_rel_pos,
        tuple(surroundings),
        tuple(bomb_info),
        can_place_bomb
    )

def act(self, game_state: dict) -> str:
    state = state_to_tuple(game_state)
    
    if state not in self.q_table:
        self.q_table[state] = np.zeros(len(ACTIONS))

    if np.random.rand() < self.epsilon:
        action = np.random.choice(ACTIONS)
    else:
        # Rauschen hinzufügen mit der gleichen Form wie die Q-Werte
        noise = np.random.randn(len(self.q_table[state])) * 0.1
        action_index = np.argmax(self.q_table[state] + noise)
        
        # Überprüfe, ob der action_index im Bereich der ACTIONS-Liste liegt
        if action_index >= len(ACTIONS) or action_index < 0:
            action_index = np.argmax(self.q_table[state])  # Fallback auf die beste bekannte Aktion
        
        action = ACTIONS[action_index]

    return action