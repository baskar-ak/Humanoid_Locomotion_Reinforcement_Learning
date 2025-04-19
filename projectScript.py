# Import necessary libraries
import os
import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters for both SAC and TD3
GAMMA = 0.99                  # Discount factor
LR = 3e-4                     # Learning rate for all optimizers
TAU = 0.01                    # Soft update rate for target networks
ALPHA = 0.2                   # Entropy regularization coefficient for SAC
POLICY_NOISE = 0.1            # Noise added to target policy during critic update (TD3)
NOISE_CLIP = 0.2              # Range to clip target policy noise (TD3)
POLICY_DELAY = 4              # Frequency of policy updates (TD3)
BUFFER_SIZE = 1000000         # Maximum size of replay buffer
BATCH_SIZE = 256              # Mini-batch size for updates
SAVE_INTERVAL = 25000         # Steps between saving model checkpoints

# Directory setup to store logs and models
log_dir = "logs"
model_dir = "models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
writer = SummaryWriter(log_dir)  # TensorBoard writer

""" Actor network for SAC """
class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Two hidden layers of 256 neurons and ReLU activation
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = self.fc(state)
        mean = self.mean(x) # Gaussian mean
        log_std = self.log_std(x).clamp(-20, 2)  # Prevents instability
        std = log_std.exp() # Log standard deviation
        return mean, std

""" Actor network for TD3 """
class TD3Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        # Two hidden layers of 256 neurons and Tanh activation for action bounds
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.max_action * self.net(state)

""" Critic network used by both SAC and TD3 """
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
         # Two hidden layers of 256 neurons and ReLU activation
        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        return self.q(torch.cat([state, action], dim=-1))

""" Experience replay buffer """
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

""" SAC update step """
def update_sac(actor, q1, q2, q1_target, q2_target, optimizer, buffer):
    # Sample a batch from buffer
    batch = buffer.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert batch to tensors
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).view(-1, 1).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones = torch.tensor(np.array(dones), dtype=torch.float32).view(-1, 1).to(device)

    # Current Q estimates
    q1_values = q1(states, actions)
    q2_values = q2(states, actions)

    # Target Q calculation
    with torch.no_grad():
        next_mean, next_std = actor(next_states)
        next_actions = next_mean + next_std * torch.randn_like(next_std).to(device)
        next_actions = torch.clamp(next_actions, -1, 1)
        next_q1 = q1_target(next_states, next_actions)
        next_q2 = q2_target(next_states, next_actions)
        next_q = torch.min(next_q1, next_q2) - ALPHA * torch.log(torch.clamp(next_std, min=1e-6)).mean(1, keepdim=True)
        target_q = rewards + GAMMA * (1 - dones) * next_q

    # Critic loss and update
    q1_loss = torch.nn.functional.mse_loss(q1_values, target_q)
    q2_loss = torch.nn.functional.mse_loss(q2_values, target_q)
    optimizer.zero_grad()
    (q1_loss + q2_loss).backward()
    optimizer.step()

    # Actor update (entropy-regularized)
    mean, std = actor(states)
    sampled_action = mean + std * torch.randn_like(std).to(device)
    actor_loss = (ALPHA * torch.log(torch.clamp(std, min=1e-6)).mean(1, keepdim=True) - q1(states, sampled_action)).mean()

    optimizer.zero_grad()
    actor_loss.backward()
    optimizer.step()

    # Soft updates for target networks
    with torch.no_grad():
        for param, target_param in zip(q1.parameters(), q1_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for param, target_param in zip(q2.parameters(), q2_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

""" TD3 update step """
def update_td3(actor, actor_target, q1, q2, q1_target, q2_target, optim_actor, optim_critic, buffer, step):
    # Sample from replay buffer
    batch = buffer.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).view(-1, 1).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones = torch.tensor(np.array(dones), dtype=torch.float32).view(-1, 1).to(device)

    with torch.no_grad():
        noise = (torch.randn_like(actions) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP).to(device)
        next_action = torch.clamp(actor_target(next_states) + noise, -1, 1)
        target_q = torch.min(q1_target(next_states, next_action), q2_target(next_states, next_action))
        target_q = rewards + GAMMA * (1 - dones) * target_q

    # Critic loss and update
    current_q1 = q1(states, actions)
    current_q2 = q2(states, actions)
    critic_loss = torch.nn.functional.mse_loss(current_q1, target_q) + torch.nn.functional.mse_loss(current_q2, target_q)

    optim_critic.zero_grad()
    critic_loss.backward()
    optim_critic.step()

    # Delayed policy update
    if step % POLICY_DELAY == 0:
        actor_loss = -q1(states, actor(states)).mean()
        optim_actor.zero_grad()
        actor_loss.backward()
        optim_actor.step()

        # Soft update of both actor and critic target networks
        with torch.no_grad():
            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            for param, target_param in zip(q1.parameters(), q1_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            for param, target_param in zip(q2.parameters(), q2_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

""" Training function for both algorithms """
def train(env, algo):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    buffer = ReplayBuffer(BUFFER_SIZE)
    step_count = 0
    episode = 0

    # Initialize models and optimizers
    if algo == "SAC":
        actor = SACActor(state_dim, action_dim).to(device)
        q1 = QNetwork(state_dim, action_dim).to(device)
        q2 = QNetwork(state_dim, action_dim).to(device)
        q1_target = QNetwork(state_dim, action_dim).to(device)
        q2_target = QNetwork(state_dim, action_dim).to(device)
        optimizer = optim.Adam(list(actor.parameters()) + list(q1.parameters()) + list(q2.parameters()), lr=LR)

    elif algo == "TD3":
        actor = TD3Actor(state_dim, action_dim, max_action).to(device)
        actor_target = TD3Actor(state_dim, action_dim, max_action).to(device)
        q1 = QNetwork(state_dim, action_dim).to(device)
        q2 = QNetwork(state_dim, action_dim).to(device)
        q1_target = QNetwork(state_dim, action_dim).to(device)
        q2_target = QNetwork(state_dim, action_dim).to(device)
        actor_target.load_state_dict(actor.state_dict())
        q1_target.load_state_dict(q1.state_dict())
        q2_target.load_state_dict(q2.state_dict())
        optim_actor = optim.Adam(actor.parameters(), lr=LR)
        optim_critic = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=LR)

    # Infinite training loop
    while True:
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)

            # Select action using current policy
            if algo == "SAC":
                mean, std = actor(state_tensor)
                action = torch.clamp(mean + std * torch.randn_like(std), -1, 1)
            elif algo == "TD3":
                action = torch.clamp(actor(state_tensor), -1, 1)
            else:
                print("Algorithm not found!")
                return

            next_state, reward, done, _, _ = env.step(action.detach().cpu().numpy())
            buffer.add((state, action.detach().cpu().numpy(), reward, next_state, done))
            state = next_state
            total_reward += reward
            step_count += 1

            # Perform learning step
            if len(buffer.buffer) > BATCH_SIZE:
                if algo == "SAC":
                    update_sac(actor, q1, q2, q1_target, q2_target, optimizer, buffer)
                elif algo == "TD3":
                    update_td3(actor, actor_target, q1, q2, q1_target, q2_target, optim_actor, optim_critic, buffer, step_count)

            # Save model
            if step_count % SAVE_INTERVAL == 0:
                torch.save(actor.state_dict(), f"{model_dir}/{algo}_{step_count}.pth")
                print(f"Model saved at step {step_count}")

        # Log episode reward
        writer.add_scalar(f"{algo}/Total_Reward", total_reward, episode)
        episode += 1

""" Testing function """
def test(env, model_path, algo):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if algo == "SAC":
        actor = SACActor(state_dim, action_dim).to(device)
    elif algo == "TD3":
        actor = TD3Actor(state_dim, action_dim, max_action).to(device)
    else:
        print("Algorithm not available!")
        return

    # Load saved weights
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()

    # Run one episode
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        action = actor(state_tensor).detach().cpu().numpy()
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
    print(f"Total reward: {total_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str)
    parser.add_argument("algo", type=str, choices=["SAC", "TD3"])
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-s", "--test", metavar="path_to_model")
    args = parser.parse_args()

    env = gym.make(args.env)

    if args.train:
        train(env, args.algo)
    elif args.test:
        if os.path.isfile(args.test):
            env = gym.make(args.env, render_mode="human")
            test(env, args.test, args.algo)
        else:
            print(f"Model file {args.test} not found.")