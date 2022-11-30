import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np

import random
from einops.layers.torch import Rearrange
from einops import rearrange

from typing import Any, Dict, Tuple, Optional
from game_mechanics import (
    ChooseMoveCheckpoint,
    ShooterEnv,
    checkpoint_model,
    choose_move_randomly,
    human_player,
    load_network,
    play_shooter,
    save_network,
)
from tqdm import tqdm

from functools import partial
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from copy import deepcopy
from functools import partial

from utils import *
from torch.utils.tensorboard import SummaryWriter
# at end run "tensorboard --logdir runs" in terminal to visualize

cpu = 'cpu'
gpu = 'cuda:0'


def choose_move(state, neural_network: nn.Module) -> int:
    probs = neural_network(state.to(device))
    probs = probs.cpu().detach().numpy()
    move = np.random.choice(range(6), p=probs)
    return int(move)


policy = nn.Sequential(
    nn.Linear(35, 1000),
    nn.LeakyReLU(),
    nn.Linear(1000, 6),
    nn.Softmax(dim=-1)
)

V = nn.Sequential(
    nn.Linear(35, 1000),
    nn.LeakyReLU(),
    nn.Linear(1000, 200),
    nn.LeakyReLU(),
    nn.Linear(200, 1)
)



gamma = 0.97
lamda = 0.6
epsilon = 0.05
entropy_coeff = 0.05
erm = EpisodeReplayMemory(gamma, lamda)
optimizer_policy = torch.optim.Adam(policy.parameters(), lr=0.001)
optimizer_value = torch.optim.Adam(V.parameters(), lr=0.005)

episodes_per_stage = 1500
n_stages = 10
batch_size = 1000
gradient_steps = 5
env = ShooterEnv(opponent_choose_move=choose_move_randomly)
tensorboard = SummaryWriter('logs/PPO_v2')
# v0 - made it work
# v1 - change value net final activation from tanh to linear
# v2 - burn-in entropy coefficient
# v3 - much smaller entropy coefficient
# v3 - add gradient steps for value function
# v4 - fix? clipping 


loss_policies = []
loss_values = []
results = []
for stage in tqdm(range(n_stages)):
    entropy_coeff = 0.0002 * stage
    if stage <= 2:
        include_barriers = False
    else:
        include_barriers = True
    if stage <= 5:
        half_sized_game = True
    else:
        half_sized_game = False
        
    env = ShooterEnv(opponent_choose_move=choose_move_randomly, #partial(choose_move, neural_network = opponent), 
             game_speed_multiplier=100_000,
             include_barriers=include_barriers,
             half_sized_game=half_sized_game)
    for episode in tqdm(range(episodes_per_stage)):
        old_observation, reward, done, info = env.reset()
        old_observation = add_features(old_observation)
#         old_value = V(old_observation)
        num_steps = 0
        while not done:
            probs = policy(old_observation)
            chosen_move = np.random.choice(range(0,6), p=probs.detach().cpu().numpy())
            observation, reward, done, info = env.step(int(chosen_move))
            observation = add_features(observation)#.to(device)
            
            erm.append({
                'old_observation': old_observation.cpu(),
                'observation': observation.cpu(),
                'reward': reward,
                'done': done,
                'chosen_move': chosen_move,
            })
            num_steps += 1
        results.append(reward)
        if len(erm) >= batch_size:
            policy.to(gpu)
            V.to(gpu)
            data = erm.pop(batch_size)
            states = data['old_observation'].to(gpu)
            dones = data['done'].to(gpu)
            rewards = data['reward'].to(gpu)
            old_values = torch.squeeze(V(states))
            with torch.no_grad():
                old_pol_dist = Categorical(policy(states))
                old_pol_probs = old_pol_dist.probs[range(batch_size), data['chosen_move'].long()]
                successor_values = torch.squeeze(V(data['observation'].to(gpu))) * (1-dones)
                
            
            gaes = gae(rewards, old_values.detach(), successor_values, dones, gamma, lamda, device=gpu)
            
            # optimize value
            loss_value = torch.nn.MSELoss()(old_values, gaes)
            optimizer_value.zero_grad()
            loss_value.backward()
            optimizer_value.step()
            
            # optimize policy
            for _ in range(gradient_steps):
                pol_dist = Categorical(policy(states))
                pol_probs = pol_dist.probs[range(batch_size), data['chosen_move'].long()]
                clipped_obj = torch.clip(pol_probs / old_pol_probs, 1 - epsilon, 1 + epsilon)
                
                ppo_obj = (
                    torch.min(clipped_obj * gaes, (pol_probs / old_pol_probs) * gaes)
                    + entropy_coeff * pol_dist.entropy()
                )
                loss_policy = -torch.sum(ppo_obj)/len(ppo_obj)
                
                optimizer_policy.zero_grad()
                loss_policy.backward()
                optimizer_policy.step()
                
            total_episode = episode + stage * episodes_per_stage
            results_last_100 = results[-100:]
            winrate_last_100 = sum(results_last_100) / len(results_last_100) / 2 + 0.5
            tensorboard.add_scalar("Policy loss", loss_policy.item(), total_episode)
            tensorboard.add_scalar("Value loss", loss_value.item(), total_episode)
            tensorboard.add_scalar('Last episode length', num_steps, total_episode)
            tensorboard.add_scalar("Entropy", pol_dist.entropy().mean().item(), total_episode)
            tensorboard.add_scalar("Winrate last 100", winrate_last_100, total_episode)
            tensorboard.add_scalar('Entropy coefficient', entropy_coeff, total_episode)
        policy.to(cpu)
        V.to(cpu)        