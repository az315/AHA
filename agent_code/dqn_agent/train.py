import numpy as np
import logging
from .callbacks import state_to_tuple, ACTIONS

# Logger für dieses Modul erstellen
logger = logging.getLogger(__name__)

def setup_training(self):
    self.transitions = []
    self.rewards = []

def reward_from_events(events):
    game_rewards = {
        'COIN_COLLECTED': 1,
        'KILLED_OPPONENT': 5,
        'MOVED_TOWARDS_COIN': 0.1,
        'AVOIDED_EXPLOSION': 0.5,
        'INVALID_ACTION': -0.5,
        'GOT_KILLED': -5,
        'KILLED_SELF': -5,
        'MOVED_TOWARDS_SAFE_AREA': 0.2,
        'MOVED_AWAY_FROM_BOMB': 0.2,
        'BOMB_DROPPED': 0.1,
        'KILLED_BY_BOMB': -2.5,
        'USELESS_BOMB': -3
    }
    reward = sum([game_rewards.get(event, 0) for event in events])
    return reward

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events):
    if old_game_state is None or new_game_state is None:
        return

    old_state = state_to_tuple(old_game_state)
    new_state = state_to_tuple(new_game_state)

    if self_action in ACTIONS:
        action_index = ACTIONS.index(self_action)
    else:
        logger.error(f"Ungültige Aktion: {self_action}")
        action_index = 0  # Default-Aktion

    if self_action == 'BOMB' and not is_bomb_strategic(old_game_state):
        events.append('USELESS_BOMB')

    reward = reward_from_events(events)
    done = False
    if 'GOT_KILLED' in events or 'KILLED_SELF' in events:
        done = True

    self.agent.remember(old_state, action_index, reward, new_state, done)

    if len(self.agent.memory.buffer) > self.agent.batch_size:
        self.agent.replay()

    logger.info(f"Game events occurred: Action - {self_action}, Reward - {reward}")

def is_bomb_strategic(game_state):
    agent_x, agent_y = game_state['self'][3]
    field = game_state['field']
    search_radius = 3

    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            x, y = agent_x + dx, agent_y + dy
            if 0 <= x < field.shape[0] and 0 <= y < field.shape[1]:
                if is_enemy_or_hazard(x, y, game_state):
                    return True
    return False

def is_enemy_or_hazard(x, y, game_state):
    # Prüfe auf Gegner
    for enemy in game_state['others']:
        if enemy[3] == (x, y):
            return True

    # Prüfe auf zerstörbare Wände
    if game_state['field'][x, y] == 1:
        return True

    return False

def end_of_round(self, last_game_state: dict, last_action: str, events):
    game_events_occurred(self, last_game_state, last_action, None, events)
    self.agent.save("dqn_model.pth")
    logger.info(f"End of round. Model saved. Events: {events}")
