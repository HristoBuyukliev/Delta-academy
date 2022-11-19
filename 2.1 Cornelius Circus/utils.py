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
        self.current_episode = []
        self.gamma = gamma
        self.lamda = lamda
        
    def append(self, datapoint):
        new_row = pd.DataFrame.from_records(datapoint, index=[uuid4()])
        if len(self.current_episode) == 0:
            self.current_episode = pd.DataFrame(columns = datapoint.keys())
        self.current_episode = pd.concat([self.current_episode, new_row])
        if datapoint['done']:
            discounted_rewards = np.array([discounted_value(self.current_episode.reward.values[i:], self.gamma) for i in range(len(self.current_episode))])
#             gaes = gae(self.current_episode.reward,
#                        self.current_episode.value.values,
#                        np.append(self.current_episode.value.values[1:], 0),
#                        self.gamma,
#                        self.lamda)
#             self.current_episode['gae'] = gaes
            self.current_episode['discounted_rewards'] = discounted_rewards
            if len(self.data) == 0:
                self.data = pd.DataFrame(columns = self.current_episode.columns)
            self.data  = pd.concat([self.data, self.current_episode])
            self.current_episode = []
            
    
    def sample(self, size):
        size = min([size, len(self.data)])
        sample = self.data.sample(size)
        return self.stack(sample)
    
    def sample_with_remove(self, size):
        size = min([size, len(self.data)])
        sample = self.data.sample(size)
        self.data = self.data.drop(sample.index)
        return self.stack(sample)
    
    def stack(self, data):
        stacked_data = {}
        for key in data.columns:
            values = data[key].values
            if type(values[0]) in [list, np.ndarray]:
                stacked_data[key] = values
            elif type(values[0]) == torch.Tensor:
                stacked_data[key] = torch.stack(tuple(values))
            elif type(values[0]) in [int, float, np.float64, bool]:
                stacked_data[key] = torch.as_tensor(values.astype('float32'), dtype=torch.float32)
                stacked_data[key] = rearrange(stacked_data[key], 'b -> 1 b')
            else:
                print(type(values[0]), type(values), key)
                raise ValueError
        return stacked_data
    
    def pop(self, size=1):
        self.data = self.data.iloc[size:]
    
    def __len__(self):
        return len(self.data)
    
def gae(rewards, values, successor_values, gamma, lamda):
    N = len(rewards)
    deltas = rewards + gamma*successor_values - values
    gamlam = gamma * lamda
    gamlam_geo_series = torch.as_tensor([gamlam**i for i in range(N)])*(1-gamlam)
    full_gamlam_matrix = torch.stack([torch.roll(gamlam_geo_series, shifts=n) for n in range(N)])
    full_gamlam_matrix = torch.triu(full_gamlam_matrix)
    # make sure it sums to one:
    # (by making the term for the last value be 1 - sum(all other terms))
    full_gamlam_matrix[:,-1] = 1 - full_gamlam_matrix[:,:-1].sum(axis=1)
    return full_gamlam_matrix @ deltas