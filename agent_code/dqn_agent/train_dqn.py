import matplotlib.pyplot as plt
import numpy as np
import logging

# Logger für dieses Modul erstellen
logger = logging.getLogger(__name__)

def plot_rewards(rewards, filename='rewards_plot.png'):
    plt.plot(rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(filename)
    plt.close()

def train_dqn(agent, env, episodes):
    rewards = []
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Replay nicht nach jedem Schritt, sondern wenn genügend Samples vorhanden sind
            if len(agent.memory.buffer) > agent.batch_size:
                agent.replay()

        rewards.append(total_reward)
        logger.info(f"Episode: {e}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

        # Alle 100 Episoden plotten
        if (e + 1) % 100 == 0:
            plot_rewards(rewards, filename=f'rewards_plot_{e+1}.png')

        # Speichere das Modell alle 100 Episoden
        if (e + 1) % 100 == 0:
            agent.save('dqn_model.pth')

        # Beispiel: Führe alle 500 Episoden eine Evaluierung durch
        if (e + 1) % 500 == 0:
            evaluate_dqn(agent, env, episodes=10)

def evaluate_dqn(agent, env, episodes=10):
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Setze Epsilon auf 0 für die Evaluierung

    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        logger.info(f"Evaluation Episode: {e+1}/{episodes}, Total Reward: {total_reward}")
        print(f"Evaluation Episode {e+1}/{episodes}, Total Reward: {total_reward}")

    agent.epsilon = original_epsilon
