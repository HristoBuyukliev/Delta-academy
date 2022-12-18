import pandas as pd
import torch
import numpy as np
from uuid import uuid4
from einops import rearrange
import torch.nn.functional as F
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


def tensorize(state):
    return torch.as_tensor(state.board, dtype=torch.float32)

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
    MCTS_policy = F.softmax(visits/optimal_temperature)
    return MCTS_policy
    
    
# softmax_visit_counts(root, verbose=True)


def choose_move_network(network, state):
    board = rearrange(tensorize(state), 'w h -> 1 w h')
    legal_moves = all_legal_moves(state.board, state.ko)
    with torch.no_grad():
        policy, value = network(board, [legal_moves])
    chosen_move = np.random.choice(range(0,82), p=policy.squeeze().numpy())
    return chosen_move

def discounted_value(rewards, gamma):
    discount_factors = np.array([gamma**i for i in range(len(rewards))])
    return (rewards*discount_factors).sum()


class EpisodeReplayMemory:
    def __init__(self, gamma, lamda):
        self.data = []
        self.gamma = gamma
        self.lamda = lamda
        
    def append(self, datapoint):
        datapoint['datapoint_id'] = str(uuid4())
        self.data.append(datapoint)            
    
    def pop(self, size):
        size = min([size, len(self.data)])
        sample = self.data[:size]
        self.data = self.data[size:]
        return self.stack(sample)
    
    def sample(self, size):
        size = min([size, len(self.data)])
        sample = self.data[:size]
        return self.stack(sample)
    
    def drop_oldest(self, size):
        size = min([size, len(self.data)])
        self.data = self.data[size:]
    
    def stack(self, data):
        stacked_data = {}
        for key in data[0].keys():
            values = [point[key] for point in data]
            if type(values[0]) in [list, np.ndarray, str]:
                stacked_data[key] = values
            elif type(values[0]) in [torch.Tensor, torch.tensor]:
                stacked_data[key] = torch.stack(tuple(values)).squeeze(dim=0)
            elif type(values[0]) in [int, float, np.float64, bool, np.int32, np.int64]:
                stacked_data[key] = torch.as_tensor(values, dtype=torch.float32)
            else:
                print(type(values[0]), type(values), key)
                raise ValueError
        return stacked_data
    
    def __len__(self):
        return len(self.data)
    
def gae(rewards, values, successor_values, dones, gamma, lamda):
    N = len(rewards)
    deltas = rewards + gamma * successor_values - values
    gamlam = gamma * lamda
    gamlam_geo_series = torch.as_tensor([gamlam**i for i in range(N)])#*(1-gamlam)
    full_gamlam_matrix = torch.stack([torch.roll(gamlam_geo_series, shifts=n) for n in range(N)])
    full_gamlam_matrix = torch.triu(full_gamlam_matrix)

    done_indexes = torch.squeeze(dones.nonzero(), dim=1).tolist()
    for terminal_index in done_indexes:
        full_gamlam_matrix[: terminal_index + 1, terminal_index + 1:] = 0

    end_index = torch.arange(N)
    for start, end in zip([-1]+done_indexes[:-1], done_indexes):
        end_index[start+1:end+1] = end
#     # make sure it sums to one:
    full_gamlam_matrix[torch.arange(N), end_index] *= 1/(1-lamda)
    return full_gamlam_matrix @ deltas


def choose_move_network(network, state):
    board = rearrange(tensorize(state), 'w h -> 1 w h')
    legal_moves = all_legal_moves(state.board, state.ko)
    with torch.no_grad():
        policy, value = network(board, [legal_moves])
        policy = F.softmax(policy, dim=-1)
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

def softmax_visit_counts(node, entropy_share=0.5, temperature=None, verbose=False):
    '''
    softmax visit counts, such that the entropy is x% of the max possible entropy
    '''
    visits = torch.as_tensor(node.child_number_visits)
    # remove illegal moves
    mask = torch.as_tensor([move not in node.legal_moves for move in range(82)])
    visits = visits.masked_fill(mask, -torch.inf)
    
#     print(node.child_number_visits)
#     print(len(node.children))
    assert (node.child_number_visits > 0).sum() != 0, 'we need some children, brah'
    
    visited_children = (node.child_number_visits > 0).sum()
    max_entropy = np.log(visited_children) # len(node.legal_moves)
    desired_bits = max_entropy*entropy_share
    if visited_children == 1:
        optimal_temperature = 1
    else:
        optimal_temperature = find_optimal_temp(visits, desired_bits)
    if temperature:
        optimal_temperature = temperature
    if verbose:
        print(f'Max entropy possible: {round(max_entropy, 2)}; we want {round(desired_bits, 2)} bits. Temperature: {optimal_temperature}')
    MCTS_policy = F.softmax(visits/optimal_temperature, dim=0)
    return MCTS_policy
