import pandas as pd
import torch
import numpy as np
from uuid import uuid4
from einops import rearrange

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
    
    def stack(self, data):
        stacked_data = {}
        for key in data[0].keys():
            values = [point[key] for point in data]
            if type(values[0]) in [list, np.ndarray, str]:
                stacked_data[key] = values
            elif type(values[0]) in [torch.Tensor, torch.tensor]:
                stacked_data[key] = torch.stack(tuple(values)).squeeze(dim=0)
            elif type(values[0]) in [int, float, np.float64, bool, np.int32]:
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

#     end_index = torch.arange(N)
#     for start, end in zip([-1]+done_indexes[:-1], done_indexes):
#         end_index[start+1:end+1] = end
#     # make sure it sums to one:
#     # (by making the term for the last value be 1 - sum(all other terms))
#     full_gamlam_matrix[torch.arange(N), end_index] += 1 - full_gamlam_matrix.sum(axis=1)
    return full_gamlam_matrix @ deltas

def angle(point1, point2):
    distance = np.linalg.norm(point1 - point2)
    if distance <= 1e-5:
        return [0, 0]
    sin = (point1[0] - point2[0]) / distance
    cos = (point1[1] - point2[1]) / distance
    if pd.isna(sin.item()) or pd.isna(cos.item()):
        print(point1, point2)
        print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        raise ValueError
    return [sin.item(), cos.item()]
    
def distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def add_features(state):
    distance_ships = distance(state[:2], state[4:6])
    angles_ships = angle(state[:2], state[4:6])

    distance_bullets1_ship2 = [distance(state[4:6], state[8:10]), distance(state[4:6], state[12:14])]
    if (state[8:10] == torch.as_tensor([-1, -1])).all().item():
        distance_bullets1_ship2[0] = 10
    if (state[12:14] == torch.as_tensor([-1, -1])).all().item():
        distance_bullets1_ship2[1] = 10
    distance_bullets1_ship2

    distance_bullets2_ship1 = [distance(state[0:2], state[16:18]), distance(state[0:2], state[20:22])]
    if (state[16:18] == torch.as_tensor([-1, -1])).all().item():
        distance_bullets1_ship2[0] = 10
    if (state[20:22] == torch.as_tensor([-1, -1])).all().item():
        distance_bullets1_ship2[1] = 10

    bullets_fired = [1,1,1,1]
    if (state[8:10] == torch.as_tensor([-1, -1])).all().item():
        bullets_fired[0] = 0
    if (state[12:14] == torch.as_tensor([-1, -1])).all().item():
        bullets_fired[1] = 0
    if (state[16:18] == torch.as_tensor([-1, -1])).all().item():
        bullets_fired[2] = 0
    if (state[20:22] == torch.as_tensor([-1, -1])).all().item():
        bullets_fired[3] = 0

    features = [distance_ships] + angles_ships + distance_bullets1_ship2 + distance_bullets2_ship1 + bullets_fired
    features = torch.as_tensor(features, dtype=torch.float32)
    return torch.cat([state, features])