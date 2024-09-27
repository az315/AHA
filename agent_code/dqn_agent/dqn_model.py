import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import logging

# Logger für dieses Modul erstellen
logging.basicConfig(filename='training.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', filemode='w')

# Gerät für Berechnungen auswählen (CPU oder GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, action_size)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return np.array(states), actions, rewards, np.array(next_states), dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(10000)
        self.gamma = 0.95
        self.epsilon = 1.0 
        self.epsilon_min = 0.2 #0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.00005
        self.batch_size = 64
        self.update_target_every = 5000
        self.steps = 0

        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)

            self.model.eval()
            with torch.no_grad():
                q_values = self.model(state)
            self.model.train()

            action = int(np.argmax(q_values.cpu().data.numpy()))
            logging.info(f"Q-Werte: {q_values.cpu().data.numpy()}")
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory.buffer) < self.batch_size:
            return


        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)
        weights = torch.FloatTensor(weights).to(device)

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        q_values = self.model(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = torch.argmax(self.model(next_states), dim=1, keepdim=True)
            next_q_value = self.target_model(next_states).gather(1, next_actions).squeeze(1)

        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)

        loss = (q_value - expected_q_value.detach()).pow(2) * weights
        prios = loss + 1e-5
        loss = loss.mean()

        logging.info(f"Loss: {loss.item()}")

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
        self.optimizer.step()

        self.memory.update_priorities(indices, prios.detach().cpu().numpy())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        logging.info(f"Epsilon nach Decay: {self.epsilon}")

        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=device))
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, name):
        torch.save(self.model.state_dict(), name)
