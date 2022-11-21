from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch import nn

from check_submission import check_submission
from game_mechanics import (
    OthelloEnv,
    choose_move_randomly,
    load_network,
    play_othello_game,
    save_network,
    get_legal_moves
)

from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from tqdm import tqdm
import random

TEAM_NAME = "Deep learners"  

pyramid = torch.ones((2,2))
pyramid = nn.functional.pad(pyramid, pad=(1,1,1,1),value=0)
pyramid = nn.functional.pad(pyramid, pad=(1,1,1,1),value=-1).to(torch.float32)

class OthelloNet(nn.Module):
    def __init__(self):
        super(OthelloNet, self).__init__()
        hidden = 20 
        # stride 1
        self.conv1 = nn.Conv2d(4,hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden*2,hidden, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden*2,1, kernel_size=1, padding=0)
        
        # stride 2
        self.conv1_s2 = nn.Conv2d(4,hidden, kernel_size=3, padding=2,dilation=2)
        self.conv2_s2 = nn.Conv2d(hidden*2,hidden, kernel_size=3, padding=2,dilation=2)
    
#         self.linear = nn.Linear(hidden*2*6*6, hidden)
    
    def forward(self, x):
        x_1a = self.conv1(x)
        x_1a = nn.functional.relu(x_1a)
        x_1b = self.conv1_s2(x)
        x_1b = nn.functional.relu(x_1b)
        
        x = torch.concat([x_1a, x_1b],dim=1)
        x_2a = self.conv2(x)
        x_2a = nn.functional.relu(x_2a)
        x_2b = self.conv2_s2(x)
        x_2b = nn.functional.relu(x_2b)

        x = torch.concat([x_2a, x_2b],dim=1)
        # add a FC layer
#         x_fc = rearrange(x, 'b c w h -> b (c w h)')
#         x_fc = self.linear(x_fc)
#         x_fc = repeat(x_fc, 'b c -> b c w h', w=6,h=6)
#         x = torch.concat([x, x_fc], dim=1)
        
        # final flatten
        x = self.conv3(x)
        x = nn.functional.tanh(x)
        x = rearrange(x, 'b 1 w h -> b w h')
        return x
    
network = OthelloNet()
    
def tensorify(np_state):
    tensor_state = torch.as_tensor(np_state, dtype=torch.float32)
    state1 = (tensor_state == 1).to(torch.float32)
    state0 = (tensor_state == 0).to(torch.float32)
    state_1 = (tensor_state == -1).to(torch.float32)
    return torch.stack([state1,state0,state_1,pyramid])
    
def greedy_move(net, state, possible_moves):
    if len(possible_moves) == 0: return None
    preds = net(rearrange(torch.as_tensor(state), 'c w h -> 1 c w h'))
    values = preds[0][np.array(possible_moves).transpose()]
    return possible_moves[values.argmax()]

def train():

    
    n_episodes = 1000
    gamma = 1
    epsilon = 0.3
    epsilon_decay = 0.999
    env = OthelloEnv()
    loss_fn = torch.nn.L1Loss()

    optim = torch.optim.AdamW(network.parameters(), lr=0.0001)
    memory = []

    N = 2000
    M = 64

    for episode in tqdm(range(n_episodes)):
        state, reward, done, info = env.reset()
        tensor_state = tensorify(state) #torch.as_tensor(state, dtype=torch.float32)
        memory_episode = []
        while not done:
            prev_state = tensor_state
            possible_moves = get_legal_moves(state)
            if len(possible_moves) == 0:
                move = None

            elif random.random() < epsilon:
                move = random.choice(possible_moves)
            else:
                move = greedy_move(network, prev_state, possible_moves)
            state, reward, done, info = env.step(move)
            tensor_state = tensorify(state)
            if move is not None:
                memory_episode.append((prev_state, reward, move, tensor_state))

            if len(memory) > N:
                memory.pop(0)

            if M < len(memory):

                random_choices = np.random.choice(range(len(memory)), size=M, replace=False)

                old_states = torch.stack([memory[idx][0] for idx in random_choices])
                states = torch.stack([memory[idx][3] for idx in random_choices])
                rewards = torch.tensor(np.array([memory[idx][1] for idx in random_choices]),
                                        dtype=torch.float32)
                moves = torch.as_tensor([memory[idx][2] for idx in random_choices], dtype=torch.long)
                old_values = network(old_states)
                old_value_moves = old_values[range(old_values.shape[0]),moves[:,0], moves[:,1]]

                with torch.no_grad():
                    new_values = network(states)
                    new_value_moves = new_values[range(new_values.shape[0]),moves[:,0], moves[:,1]]
                loss = loss_fn(old_value_moves, rewards + gamma * new_value_moves)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 50.0)
                optim.step()
        num_steps = len(memory_episode)
        for idx, step in enumerate(memory_episode):
            discounted_reward = reward * gamma**(num_steps-idx)
            memory_episode[idx] = (step[0], discounted_reward, step[2], step[3])
        memory = memory + memory_episode

        epsilon *= epsilon_decay
    return network

def choose_move(state: Any, network) -> Optional[Tuple[int, int]]:
    """The arguments in play_connect_4_game() require functions that only take the state as
    input.

    This converts choose_move() to that format.
    """
#     state = torch.as_tensor(state, dtype=torch.float32)
    possible_moves = get_legal_moves(state)
    tensor_state = tensorify(state)
    return greedy_move(network, tensor_state, possible_moves)


if __name__ == "__main__":

    ## Example workflow, feel free to edit this! ###
    my_network = train()
    save_network(my_network, TEAM_NAME)

    check_submission(
    )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    my_network = load_network(TEAM_NAME)

    # Code below plays a single game of othello against a random
    #  opponent, think about how you might want to adapt this to
    #  test the performance of your algorithm.
    def choose_move_no_value_fn(state: Any) -> Optional[Tuple[int, int]]:
        """The arguments in play_connect_4_game() require functions that only take the state as
        input.

        This converts choose_move() to that format.
        """
        return choose_move(state, my_network)

    outcomes = {}
    for _ in tqdm(range(1000)):
        reward = play_othello_game(
            your_choose_move=choose_move_no_value_fn,
            opponent_choose_move=choose_move_randomly,
            game_speed_multiplier=10000000000000000,
            verbose=False,
        )
        outcomes[reward] = outcomes.get(reward, 0) + 1

    print(outcomes)
