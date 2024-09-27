import numpy as np
import logging
from .callbacks import state_to_tuple, ACTIONS

# Logger für dieses Modul erstellen
logger = logging.getLogger(__name__)

def setup_training(self):
    self.transitions = []
    self.rewards = []

def adjust_rewards(episode, game_rewards):
    if episode < 500:
        game_rewards['COIN_COLLECTED'] = 2
    else:
        game_rewards['COIN_COLLECTED'] = 1

def reward_from_events(events, is_last_coin_collected, agent_health, opponent_killed, agent_too_close_to_bomb, new_area_unlocked, agent_avoided_dead_end, consecutive_explosion_avoidance, bomb_dropped_and_survived, multiple_crates_destroyed, episode):
    game_rewards = {
        'COIN_COLLECTED': 1.5,
        'KILLED_OPPONENT': 8,
        'MOVED_TOWARDS_COIN': 0.2,
        'AVOIDED_EXPLOSION': 2,
        'INVALID_ACTION': -1,
        'GOT_KILLED': -10,
        'KILLED_SELF': -10,
        'MOVED_TOWARDS_SAFE_AREA': 0.3,
        'MOVED_AWAY_FROM_BOMB': 0.5,
        'BOMB_DROPPED': 0.1,
        'KILLED_BY_BOMB': -5,
        'USELESS_BOMB': -5,
        'WAIT': -0.2
    }
    adjust_rewards(episode, game_rewards)

    # Zusätzliche Kontexte für adaptive Belohnungen
    if is_last_coin_collected:
        game_rewards['COIN_COLLECTED'] += 3  # Höhere Belohnung für letzte Münze
    if agent_health < 2 and opponent_killed:
        game_rewards['KILLED_OPPONENT'] += 10  # Sehr hohe Belohnung für Risiko-Kills

    if agent_too_close_to_bomb:
        game_rewards['AVOIDED_EXPLOSION'] -= 1  # Bestrafung für riskantes Verhalten
    if new_area_unlocked:
        game_rewards['BOMB_DROPPED'] += 0.5  # Belohnung für das Freilegen neuer Bereiche
    if agent_avoided_dead_end:
        game_rewards['MOVED_TOWARDS_SAFE_AREA'] += 0.3  # Belohnung für das Vermeiden von Sackgassen

    if consecutive_explosion_avoidance > 3:
        game_rewards['AVOIDED_EXPLOSION'] += 2  # Bonus für das wiederholte Vermeiden von Explosionen
    if bomb_dropped_and_survived:
        game_rewards['BOMB_DROPPED'] += 1  # Bonus für strategisches Bombenlegen
    if multiple_crates_destroyed:
        game_rewards['BOMB_DROPPED'] += 3 
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
        action_index = 0

    is_last_coin_collected = 'COIN_COLLECTED' in events and len(new_game_state['coins']) == 0
    agent_health = old_game_state['self'][2]
    opponent_killed = 'KILLED_OPPONENT' in events
    agent_too_close_to_bomb = agent_in_bomb_range(old_game_state)
    new_area_unlocked = is_new_area_unlocked(old_game_state, new_game_state)
    agent_avoided_dead_end = avoided_dead_end(old_game_state, new_game_state)
    consecutive_explosion_avoidance = getattr(self, 'consecutive_explosion_avoidance', 0)
    if 'AVOIDED_EXPLOSION' in events:
        self.consecutive_explosion_avoidance += 1
    else:
        self.consecutive_explosion_avoidance = 0
    bomb_dropped_and_survived = 'BOMB_DROPPED' in events and 'GOT_KILLED' not in events
    multiple_crates_destroyed = events.count('CRATE_DESTROYED') > 1

    # Sicherstellen, dass das Attribut 'episode' existiert
    if not hasattr(self, 'episode'):
        self.episode = 0  # oder ein anderer Standardwert

    reward = reward_from_events(events, is_last_coin_collected, agent_health, opponent_killed, agent_too_close_to_bomb, new_area_unlocked, agent_avoided_dead_end, consecutive_explosion_avoidance, bomb_dropped_and_survived, multiple_crates_destroyed, episode=self.episode)

    done = False
    if 'GOT_KILLED' in events or 'KILLED_SELF' in events:
        done = True

    self.agent.remember(old_state, action_index, reward, new_state, done)

    if len(self.agent.memory.buffer) > self.agent.batch_size:
        self.agent.replay()

    logger.info(f"Game events occurred: Action - {self_action}, Reward - {reward}")

def agent_in_bomb_range(game_state):
    agent_x, agent_y = game_state['self'][3]
    bombs = game_state['bombs']
    explosion_radius = 3  # Beispiel-Explosionsradius

    for bomb, _ in bombs:
        bomb_x, bomb_y = bomb
        if abs(agent_x - bomb_x) <= explosion_radius and abs(agent_y - bomb_y) <= explosion_radius:
            return True  # Der Agent ist im Explosionsradius einer Bombe

    return False

def is_new_area_unlocked(old_game_state, new_game_state):
    # Überprüfen, ob der Schlüssel 'explored' im old_game_state vorhanden ist
    if 'explored' in old_game_state:
        old_explored = np.count_nonzero(old_game_state['explored'])
    else:
        old_explored = 0  # Standardwert, wenn der Schlüssel nicht vorhanden ist

    # Überprüfen, ob der Schlüssel 'explored' im new_game_state vorhanden ist
    if 'explored' in new_game_state:
        new_explored = np.count_nonzero(new_game_state['explored'])
    else:
        new_explored = 0  # Standardwert, wenn der Schlüssel nicht vorhanden ist

    return new_explored > old_explored

def avoided_dead_end(old_game_state, new_game_state):
    old_position = old_game_state['self'][3]
    new_position = new_game_state['self'][3]
    old_surroundings = get_surroundings(old_game_state, old_position)
    new_surroundings = get_surroundings(new_game_state, new_position)

    if old_surroundings['walls'] > new_surroundings['walls']:
        return True

    return False
def get_surroundings(game_state, position):
    x, y = position
    field = game_state['field']
    walls = 0

    # Zähle die Anzahl der Wände um den Agenten herum
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if field[x + dx, y + dy] == -1:  # -1 könnte eine Wand darstellen
            walls += 1

    return {'walls': walls}

def end_of_round(self, last_game_state: dict, last_action: str, events):
    game_events_occurred(self, last_game_state, last_action, None, events)
    self.agent.save("dqn_model.pth")
    logger.info(f"End of round. Model saved. Events: {events}")

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
    for enemy in game_state['others']:
        if enemy[3] == (x, y):
            return True

    if game_state['field'][x, y] == 1:
        return True

    return False

def end_of_round(self, last_game_state: dict, last_action: str, events):
    game_events_occurred(self, last_game_state, last_action, None, events)
    self.agent.save("dqn_model.pth")