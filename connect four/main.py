import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from einops import reduce, rearrange
from torchsummary import summary

from einops import rearrange, reduce
from scipy import ndimage
from functools import partial
# from check_submission import check_submission
from game_mechanics import (
    Connect4Env,
    choose_move_randomly,
    get_empty_board,
    get_piece_longest_line_length,
    get_top_piece_row_index,
    has_won,
    is_column_full,
    load_dictionary,
    place_piece,
    play_connect_4_game,
    save_dictionary,
)
from tqdm import tqdm 
from copy import deepcopy

TEAM_NAME = "WIP2"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"

random_feature_vectors = np.random.randint(-1,2,(6,8,5))


# def to_feature_vector(board: np.ndarray) -> Tuple:
#     """
#     TODO: Write this function to convert the state to a feature vector.

#     We suggest you use functions in game_mechanics.py to make a handcrafted
#      feature vector based on the state of the board.

#     Feature vectors are covered in Tutorial 6 (don't need 4 or 5 to do 6!)

#     Args:
#         state: board state as a np array. 1's are your pieces. -1's your
#                 opponent's pieces & 0's are empty.

#     Returns: the feature for this state, as designed by you.
#     """
#     mult = (rearrange(board, 'w h -> w h 1')*random_feature_vectors)
#     random_features = (reduce(mult, 'w h f -> f', 'sum') > 0).astype(int)
#     return random_features


def to_feature_vector(board: np.ndarray) -> Tuple:

    ## 1d filter and target value. 

    x = 100 # some random big number which only works with empty fields
    filters = [
        ([x,1,1,1,x], 3),# double-open triplet
        ([x,-1,-1,-1,x], 3),
        ([x,1,1,x], 2), # double-open twople
        ([x,-1,-1,x], 2),
        ([1,x,1], 2), # single-gap triplet
        ([-1,x,-1], 2),
        ([1,1,1,x], 3),# single-open triplet, right
        ([-1,-1,-1,x], 3),
        ([x,1,1,1], 3),# single-open triplet, left
        ([x,-1,-1,-1], 3),
        ([0,1,1,1,1],4),
        ([0,1,1,1,1],-4)
    ]
    features = []
    for filter_, target in filters:
        features.append(int((ndimage.convolve1d(board, filter_, cval=13, mode='constant') == target).any()))
    bt = board.transpose()
    for filter_, target in filters:
        features.append(int((ndimage.convolve1d(bt, filter_, cval=13, mode='constant') == target).any()))
    return np.array(features)

# x = 100 # some random big number which only works with zero
# filters = [
#     ([x,1,1,1,x], 3),# double-open triplet
#     ([x,1,1,x,0], 2), # double-open twople
# #     ([0,1,x,1,0], 2), # single-gap triplet
#     ([0,1,1,1,x], 3),# single-open triplet, right
#     ([x,1,1,1,0], 3),# single-open triplet, left
# ]
# filter_matrix = np.array([f[0] for f in filters])
# filter_tensor = torch.Tensor(rearrange(filter_matrix, ('in kernel -> in 1 kernel')))
# targets = np.array([f[1] for f in filters])
# targets = rearrange(torch.Tensor(targets), 'f -> 1 f 1')
# layer = nn.Conv1d(in_channels=1, out_channels=len(filters), bias=False, kernel_size=5)

# def to_feature_vector(board: np.ndarray):
#     board_tensor = nn.functional.pad(torch.Tensor(board.reshape(6,1,8)), (2,2), mode='constant', value=100)
#     filtered = nn.functional.conv1d(board_tensor, filter_tensor, bias=None, padding='valid')
#     features_me_hor  = (filtered ==  targets).any(axis=0).any(axis=1).detach().numpy().astype(int)
#     features_him_hor = (filtered == -targets).any(axis=0).any(axis=1).detach().numpy().astype(int)
    
#     board_tensor = nn.functional.pad(torch.Tensor(board.transpose().reshape(8,1,6)), (2,2), mode='constant', value=100)
#     filtered = nn.functional.conv1d(board_tensor, filter_tensor, bias=None, padding='valid')
#     features_me_ver  = (filtered ==  targets).any(axis=0).any(axis=1).detach().numpy().astype(int)
#     features_him_ver = (filtered == -targets).any(axis=0).any(axis=1).detach().numpy().astype(int)
#     return np.concatenate([features_me_hor, features_me_ver, features_him_hor, features_him_ver])

# #     filtered = nn.functional.conv1d(torch.Tensor(board.reshape(6,1,8)), filter_tensor, bias=None, padding='same')
# #     features_horz = (filtered == targets).sum(axis=0).sum(axis=1).clamp(-3,3).detach().numpy().astype(int)
# #     filtered = nn.functional.conv1d(torch.Tensor(board.transpose().reshape(8,1,6)), filter_tensor, bias=None, padding='same')
# #     features_vert = (filtered == targets).sum(axis=0).sum(axis=1).clamp(-3,3).detach().numpy().astype(int)
# #     return np.concatenate([features_horz, features_vert])


def train() -> Dict:
    """
    TODO: Write this function to train your algorithm.

    Returns:
        Value function dictionary used by your agent. You can
         structure this how you like, but choose_move() expects
         {feature_vector: value}. If you structure it another
         way, you'll have to tweak choose_move().
    """
    size = 4
    n_episodes = 10**size
    alpha = 0.2
    epsilon = 0.1
    alpha_decay = 1 - 0.1**size
    epsilon_decay = 1 - 0.1**size
    gamma = 0.9
    value_fn = {}
    score = {'won': 0, 'total': 1}
    env = Connect4Env()
    for episode in tqdm(range(n_episodes)):
        state, reward, done, info = env.reset(0)
        while not done:
            old_features = to_feature_vector(state).tobytes()
            old_reward = reward
            move = choose_move(state, value_fn, epsilon=epsilon)
            state, reward, done, info = env.step(move, 0)
            features = to_feature_vector(state).tobytes()
            old_evaluation = value_fn.get(old_features,0)
            new_evaluation = value_fn.get(features,0)
            value_fn[old_features] = old_evaluation*(1-alpha) + alpha*(old_reward + gamma*new_evaluation)
        epsilon *= epsilon_decay
        alpha *= alpha_decay
        value_fn[features] = (1-alpha)*value_fn.get(features, 0) + alpha*reward
#         score['total'] += 1
#         if reward == 1:
#             score['won'] += 1
#         if score['won'] / score['total'] > 0.6 and score['total'] > 20:
#             print(score)
#             print(f'60% winrate at reached at episode {episode}, upgrading opponent...')
#             env._opponent_choose_move = partial(choose_move, value_function=deepcopy(value_fn), epsilon=0)
#             score = {'won': 0, 'total': 1}
    return value_fn

    
    


def choose_move(state: np.ndarray,
                value_function: Dict,
                verbose: bool = False, 
                epsilon: int = 0) -> int:
    """Called during competitive play. It acts greedily given current state of the board and value
    function dictionary. It returns a single move to play.

    Args:
        state: State of the board as a np array. Your pieces are
                1's, the opponent's are -1's and empty are 0's.
        value_function: The dictionary output by train().
        verbose: Whether to print debugging information to console.

    Returns:
        position (int): The column you want to place your counter
                        into (an integer 0 -> 7), where 0 is the far
                        left column and 7 is the far right column.
    """
    values = []
    not_full_cols = [
        col for col in range(state.shape[1]) if not is_column_full(state, col)
    ]
    
    if random.random() < epsilon:
        return random.choice(not_full_cols)

    for not_full_col in not_full_cols:
        # Do 1-step lookahead and compare values of successor states
        state_copy = state.copy()
        place_piece(board=state_copy, column_idx=not_full_col, player=1)

        # Get the feature vector associated with the successor state
        features = to_feature_vector(state_copy).tobytes()
        if verbose:
            print(
                "Column index:",
                not_full_col,
                "Feature vector:",
                features,
                "Value:",
                value_function.get(features, 0),
            )

        # Add the value of the sucessor state to the values list
        values.append(value_function.get(features, 0))

    # Pick randomly between actions that have successor states with the maximum value
    max_value = max(values)
    value_indices = [
        index for index, value in enumerate(values) if value == max_value
    ]
    value_index = random.choice(value_indices)
    return not_full_cols[value_index]


if __name__ == "__main__":

    ## Example workflow, feel free to edit this! ###
    my_value_fn = train()
    save_dictionary(my_value_fn, TEAM_NAME)

#     check_submission(
#         TEAM_NAME
#     )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    my_value_fn = load_dictionary(TEAM_NAME)
    print(len(my_value_fn))

    results = {}
    for game in tqdm(range(1_000)):
        result = play_connect_4_game(
            your_choose_move=partial(choose_move, value_function=my_value_fn, epsilon=0),
            opponent_choose_move=choose_move_randomly,
            game_speed_multiplier=100000000,
            render=False,
            verbose=False,
        )
        results[result] = results.get(result, 0) + 1
    print(results)