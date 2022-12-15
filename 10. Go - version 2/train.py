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
from utils import EpisodeReplayMemory
from torch.utils.tensorboard import SummaryWriter


tensorboard = SummaryWriter('logs/AZ_v0')
env = GoEnv(choose_move_randomly)
state, reward, done, info = env.reset() 
net = AlphaGoZeroBatch(n_residual_blocks=architecture_settings['n_residual_blocks'], 
                       block_width=architecture_settings['block_width'])
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


def choose_move_network(network, state):
    board = rearrange(tensorize(state), 'w h -> 1 w h')
    legal_moves = all_legal_moves(state.board, state.ko)
    with torch.no_grad():
        policy, value = network(board, [legal_moves])
    chosen_move = np.random.choice(range(0,82), p=policy.squeeze().numpy())
    return chosen_move

def entropy(values, T=1):
    probs = F.softmax(values/T, dim=0)
    nonzero_probs = probs[probs != 0]
    return -(torch.log(nonzero_probs)*nonzero_probs).sum()

def find_optimal_temp(values, desired_bits, temps_to_check = np.logspace(-2, 2, 1000)):
    if len(temps_to_check) == 1: return temps_to_check[0]
    mid_index = round(len(temps_to_check)/2)
    mid_ent1 = entropy(values, T=temps_to_check[mid_index-1])
    mid_ent2 = entropy(values, T=temps_to_check[mid_index])
    if np.isnan(mid_ent1):
        return find_optimal_temp(values, desired_bits, temps_to_check[mid_index:])
    if np.isnan(mid_ent2):
        return find_optimal_temp(values, desired_bits, temps_to_check[:mid_index])
    if mid_ent2 < desired_bits:
        return find_optimal_temp(values, desired_bits, temps_to_check[mid_index:])
    elif mid_ent1 > desired_bits:
        return find_optimal_temp(values, desired_bits, temps_to_check[:mid_index])
    else:
        return temps_to_check[mid_index-1]

def softmax_visit_counts(node, entropy_share=0.5, verbose=False):
    '''
    softmax visit counts, such that the entropy is x% of the max possible entropy
    '''
    visits = torch.as_tensor(node.child_number_visits)
    # remove illegal moves
    mask = torch.as_tensor([move not in node.legal_moves for move in range(82)])
    visits = visits.masked_fill(mask, -torch.inf)
    
    visited_children = (node.child_number_visits > 0).sum()
    max_entropy = np.log(visited_children) # len(node.legal_moves)
    desired_bits = max_entropy*entropy_share
    if visited_children == 1:
        optimal_temperature = 1
    else:
        optimal_temperature = find_optimal_temp(visits, desired_bits)
    if verbose:
        print(f'Max entropy possible: {round(max_entropy, 2)}; we want {round(desired_bits, 2)} bits. Temperature: {optimal_temperature}')
    MCTS_policy = F.softmax(visits/optimal_temperature, dim=0)
    return MCTS_policy


erm = EpisodeReplayMemory(gamma=1, lamda=1)

start = datetime.now()
for episode in tqdm(range(3000)):
    state, reward, done, info = env.reset() 
    while not done:
        train_this_move = False
        if random.random() > 0.75:
            train_this_move = True
            expanded_root = UCT_search(state, 400, network=net)
        else:
            expanded_root = UCT_search(state, 20, network=net)
        
        MCTS_policy = softmax_visit_counts(expanded_root, entropy_share=0.4, verbose=False)
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
    if len(erm) > 1024 and (episode % 10 == 0):
        for batch_number in tqdm(range(20)):
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
        
    if len(erm) > 10_000:
        print('pruning EpisodeReplayMemory...')
        erm.drop_oldest(len(erm) - 10_000)
        

            
    print(f"We've collected {len(erm)} datapoints for {datetime.now() - start} seconds: {len(erm)/(datetime.now() - start).seconds} per second")