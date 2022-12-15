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

    end_index = torch.arange(N)
    for start, end in zip([-1]+done_indexes[:-1], done_indexes):
        end_index[start+1:end+1] = end
#     # make sure it sums to one:
    full_gamlam_matrix[torch.arange(N), end_index] *= 1/(1-lamda)
    return full_gamlam_matrix @ deltas