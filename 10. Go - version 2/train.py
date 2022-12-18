import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random
from einops.layers.torch import Rearrange
from einops import rearrange

from typing import Any, Dict, Tuple, Optional
from game_mechanics import (
    State,
    all_legal_moves,
    choose_move_randomly,
    human_player,
    is_terminal,
    load_pkl,
    play_go,
    reward_function,
    save_pkl,
    transition_function,
)
from game_mechanics.go_env import GoEnv
from tqdm import tqdm

from functools import partial
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from network import *
from configs.v0_PPO_MCTS import *
from MCTS import *
from utils import *
from utils import EpisodeReplayMemory
from torch.utils.tensorboard import SummaryWriter


tensorboard = SummaryWriter('logs/AZ_v0')
env = GoEnv(choose_move_randomly)
state, reward, done, info = env.reset() 
net = AlphaGoZeroBatch(n_residual_blocks=architecture_settings['n_residual_blocks'], 
                       block_width=architecture_settings['block_width'])
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


erm = EpisodeReplayMemory(gamma=1, lamda=1)

start = datetime.now()
for episode in tqdm(range(3000)):
    state, reward, done, info = env.reset() 
    while not done:
        train_this_move = False
        if random.random() > 0.75:
            train_this_move = True
#             print('searching 100 states')
            expanded_root = UCT_search(state, 200, batch_size=16, network=net)
        else:
#             print('searching 20 states')
            expanded_root = UCT_search(state, 20, batch_size=2, network=net)
        
        MCTS_policy = softmax_visit_counts(expanded_root, temperature=1, verbose=False)
        chosen_move = np.random.choice(range(0,82), p=MCTS_policy.numpy())
        erm.append({'board': tensorize(state),
                    'to_play': state.to_play,
                    'MCTS_policy': MCTS_policy,
                    'reward': reward,
                    'done': done,
                    'MCTS_value': expanded_root.total_value/expanded_root.number_visits,
                    'chosen_move': chosen_move,
                    'legal_moves': expanded_root.legal_moves})
        state, reward, done, info = env.step(chosen_move)
    if len(erm) > 128:
        for batch_number in range(5):
            data = erm.sample(128)
            policy, value = net(data['board'], data['legal_moves'])
            optimizer.zero_grad()
            loss_policy = F.cross_entropy(policy, data['MCTS_policy'])
            loss_value = F.smooth_l1_loss(value, data['MCTS_value'])
            loss = loss_policy + loss_value
            loss.backward()
            optimizer.step()
        # play 100 games vs. random opponent:
        total_score = 0
        for game in range(100):
            total_score += play_go(your_choose_move = partial(choose_move_network, network=net),
                                   opponent_choose_move = choose_move_randomly,
                                   game_speed_multiplier = 100000,
                                   render=False)
        winrate = total_score / 200 + 0.5
        tensorboard.add_scalar("Policy loss", loss_policy.item(), episode)
        tensorboard.add_scalar("Value loss", loss_value.item(), episode)
        tensorboard.add_scalar("Winrate 100", winrate, episode)

    if len(erm) > 100:
        print('pruning EpisodeReplayMemory...')
        erm.drop_oldest(len(erm) - episode * 40)
        

            
    print(f"We've collected {len(erm)} datapoints for {datetime.now() - start} seconds: {len(erm)/(datetime.now() - start).seconds} per second")