import numpy as np
import os
import logging
from .dqn_model import DQNAgent

# Aktionen definieren
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_SIZE = len(ACTIONS)

# Logger für dieses Modul erstellen
logger = logging.getLogger(__name__)

def setup(self):
    STATE_SIZE = 8  # Angepasste Zustandsgröße nach Erweiterung der Features
    self.agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    try:
        if os.path.isfile("dqn_model.pth"):
            self.agent.load("dqn_model.pth")
            logger.info("DQN-Modell erfolgreich geladen.")
        else:
            logger.info("Kein vorhandenes Modell gefunden, starte mit einem neuen.")
    except Exception as e:
        logger.error(f"Fehler beim Laden des Modells: {e}")
        logger.info("Starte mit einem neuen Modell.")

def act(self, game_state: dict) -> str:
    state = state_to_tuple(game_state)
    action_index = self.agent.act(state)
    if 0 <= action_index < len(ACTIONS):
        logger.info(f"Action taken: {ACTIONS[action_index]} for state: {state}")
        return ACTIONS[action_index]
    else:
        logger.error(f"Ungültiger Aktionsindex: {action_index}")
        return 'WAIT'  # Fallback-Aktion

def state_to_tuple(game_state: dict) -> np.array:
    """
    Konvertiert den Spielzustand in einen Feature-Vektor.
    """
    if game_state is None:
        return np.zeros(10)  # Rückgabe eines Null-Vektors, wenn kein Zustand vorhanden ist

    own_position = np.array(game_state['self'][3])
    field = game_state['field']
    coins = game_state['coins']
    bombs = game_state['bombs']
    others = [agent[3] for agent in game_state['others']]

    # Relative Position zur nächsten Münze
    if coins:
        coin_distances = [np.linalg.norm(np.array(coin) - own_position) for coin in coins]
        nearest_coin = coins[np.argmin(coin_distances)]
        coin_rel_pos = nearest_coin - own_position
    else:
        coin_rel_pos = np.array([0, 0])

    # Relative Position zur nächsten Bombe
    if bombs:
        bomb_distances = [np.linalg.norm(np.array(bomb[0]) - own_position) for bomb in bombs]
        nearest_bomb = bombs[np.argmin(bomb_distances)][0]
        bomb_rel_pos = nearest_bomb - own_position
    else:
        bomb_rel_pos = np.array([0, 0])

    # Informationen über umliegende Felder (Wände, freie Felder)
    surroundings = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # LEFT, RIGHT, UP, DOWN
    for dx, dy in directions:
        x, y = own_position + np.array([dx, dy])
        if 0 <= x < field.shape[0] and 0 <= y < field.shape[1]:
            surroundings.append(field[int(x), int(y)])
        else:
            surroundings.append(-1)  # Markiere Positionen außerhalb des Spielfelds

    # Normalisierung der relativen Positionen
    max_distance = max(field.shape)
    coin_rel_pos = coin_rel_pos / max_distance
    bomb_rel_pos = bomb_rel_pos / max_distance

    # Feature-Vektor erstellen
    features = np.concatenate((coin_rel_pos, bomb_rel_pos, surroundings))

    # Zusätzliche Features können hier hinzugefügt werden

    return features
