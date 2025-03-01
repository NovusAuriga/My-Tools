import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from colorama import init, Fore, Style
import os
import glob
from torch.utils.tensorboard import SummaryWriter

# Initialize colorama
init()

# Game settings
TABLE_WIDTH = 25
TABLE_HEIGHT = 25
UP, DOWN, RIGHT, LEFT = 0, 1, 2, 3
DIRECTIONS_VECT = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int32)

# DQN Hyperparameters
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999
MEMORY_SIZE = 50000
BATCH_SIZE = 32
LR = 0.0001
SAVE_INTERVAL = 200
TARGET_UPDATE_FREQ = 200

# Neural Network for DQN
class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.fc(x)

# Game Environment
class SnakeGame:
    def __init__(self):
        self.max_dist = np.sqrt(TABLE_WIDTH ** 2 + TABLE_HEIGHT ** 2)  # Max possible distance
        self.reset()  # Call reset after max_dist is defined
    
    def reset(self):
        self.table = np.zeros((TABLE_HEIGHT, TABLE_WIDTH), dtype=np.int32)
        self.snake = [(TABLE_HEIGHT // 2, TABLE_WIDTH // 2)]
        self.food = self._spawn_food()
        self.direction = LEFT
        self.table[self.snake[0]] = 1
        self.table[self.food] = 2
        self.food_eaten = 0
        return self._get_state()
    
    def _spawn_food(self):
        while True:
            food = (np.random.randint(0, TABLE_HEIGHT), np.random.randint(0, TABLE_WIDTH))
            if food not in self.snake:
                return food
    
    def _get_state(self):
        head_y, head_x = self.snake[0]
        food_y, food_x = self.food
        dist_to_food = np.sqrt((head_y - food_y) ** 2 + (head_x - food_x) ** 2) / self.max_dist  # Normalize
        food_up = float(food_y < head_y)
        food_down = float(food_y > head_y)
        food_left = float(food_x < head_x)
        food_right = float(food_x > head_x)
        wall_up = float(head_y == 0)
        wall_down = float(head_y == TABLE_HEIGHT - 1)
        wall_left = float(head_x == 0)
        wall_right = float(head_x == TABLE_WIDTH - 1)
        return np.array([
            head_y / TABLE_HEIGHT, head_x / TABLE_WIDTH,
            food_y / TABLE_HEIGHT, food_x / TABLE_WIDTH,
            self.direction / 3.0,
            dist_to_food, food_up, food_down, food_left, food_right,
            wall_up, wall_down, wall_left, wall_right
        ], dtype=np.float32)
    
    def step(self, action):
        head_y, head_x = self.snake[0]
        new_head = (head_y + DIRECTIONS_VECT[action][0], head_x + DIRECTIONS_VECT[action][1])
        reward = -0.01
        done = False
        
        if (new_head[0] >= TABLE_HEIGHT or new_head[1] >= TABLE_WIDTH or 
            new_head[0] < 0 or new_head[1] < 0):
            reward = -0.5
            done = True
            return self._get_state(), reward, done

        if new_head in self.snake:
            reward = - 5
            done = True
            return self._get_state(), reward, done
        
        self.snake.insert(0, new_head)
        if new_head == self.food:
            reward = 2.0
            self.food_eaten += 1
            self.food = self._spawn_food()
        else:
            self.snake.pop()
        
        self.direction = action
        self.table.fill(0)
        for pos in self.snake:
            self.table[pos] = 1
        self.table[self.food] = 2
        
        return self._get_state(), reward, done
    
    def render(self):
        for row in self.table:
            for cell in row:
                if cell == 1:
                    print(f"{Fore.GREEN}{cell}{Style.RESET_ALL}", end=" ")
                elif cell == 2:
                    print(f"{Fore.RED}{cell}{Style.RESET_ALL}", end=" ")
                else:
                    print(cell, end=" ")
            print()
        print("\n")

# Training Loop
def train_dqn():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    writer = SummaryWriter(log_dir="runs/snake_dqn")

    game = SnakeGame()
    model = DQNetwork(input_size=14, output_size=4).to(device)
    target_model = DQNetwork(input_size=14, output_size=4).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=LR)
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = EPSILON
    start_episode = 0

    if not os.path.exists("models"):
        os.makedirs("models")

    checkpoint_files = glob.glob("models/snake_dqn_episode_*.pth")
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        if isinstance(checkpoint, dict):
            start_episode = checkpoint.get('episode', 0)
            model.load_state_dict(checkpoint['model_state_dict'])
            target_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', optimizer.state_dict()))
            epsilon = checkpoint.get('epsilon', EPSILON)
            print(f"Loaded full checkpoint from {latest_checkpoint}, resuming from episode {start_episode}, epsilon {epsilon}")
        else:
            start_episode = int(latest_checkpoint.split('_')[-1].split('.')[0])
            model.load_state_dict(checkpoint)
            target_model.load_state_dict(checkpoint)
            print(f"Loaded legacy checkpoint from {latest_checkpoint}, resuming from episode {start_episode}, epsilon reset to {epsilon}")
    else:
        print("No checkpoint found, starting training from scratch.")

    episode = start_episode
    
    while True:
        state = game.reset()
        total_reward = 0
        done = False
        episode_loss = 0
        steps = 0
        
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(device)
                    q_values = model(state_tensor)
                    action = torch.argmax(q_values).item()
            
            next_state, reward, done = game.step(action)
            reward = np.clip(reward, -1.0, 1.0)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            steps += 1
            
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)
                
                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_model(next_states).max(1)[0]
                targets = rewards + (1 - dones) * GAMMA * next_q_values
                
                loss = nn.MSELoss()(q_values, targets.detach())
                optimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                loss.backward()
                optimizer.step()
                episode_loss += loss.item()
                
                if steps % TARGET_UPDATE_FREQ == 0:
                    target_model.load_state_dict(model.state_dict())
            
            game.render()
            time.sleep(0.05)
        
        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY
        
        avg_loss = episode_loss / max(steps, 1) if episode_loss > 0 else 0
        
        writer.add_scalar("Reward/Total", total_reward, episode)
        writer.add_scalar("Loss/Average", avg_loss, episode)
        writer.add_scalar("Epsilon", epsilon, episode)
        writer.add_scalar("Steps", steps, episode)
        writer.add_scalar("Food Eaten", game.food_eaten, episode)

  

        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, "
              f"Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.3f}, Steps: {steps}, "
              f"Food Eaten: {game.food_eaten}")
        
        if episode % SAVE_INTERVAL == 0 and episode > 0:
            save_path = f"models/snake_dqn_episode_{episode}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'episode': episode
            }, save_path)
            print(f"Model saved to {save_path}")
        
        episode += 1

if __name__ == "__main__":
    try:
        train_dqn()
    except KeyboardInterrupt:
        print("\nStopping...")
