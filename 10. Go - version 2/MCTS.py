import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from game_mechanics import (
    all_legal_moves, 
    transition_function,
    reward_function, 
    is_terminal
)
import math
import collections
import random
from utils import tensorize

# def UCT_search(state, num_reads, network):
#     root = UCTNode(state, move=0, parent=DummyNode())
#     for _ in range(num_reads):
#         leaf = root.select_leaf()
#         board = tensorize(leaf.state)
#         legal_moves = leaf.legal_moves #all_legal_moves(leaf.state.board, leaf.state.ko)
#         with torch.no_grad():
#             child_priors, value_estimate = network(rearrange(board, 'w h -> 1 w h'), [legal_moves])
#             leaf.expand(child_priors.squeeze())
#             leaf.backup(value_estimate.squeeze())
#     return root

def UCT_search(state, num_reads, batch_size, network):
    root = UCTNode(state, move=0, parent=DummyNode())
    root.expand(child_priors=torch.zeros(82))
    all_leafs = []
    while num_reads > 0:
        leafs = []
        boards = []
        legal_moves_sets = []
        current_batch_size = min(batch_size, num_reads)
        for _ in range(current_batch_size):
            leaf = root.select_leaf()
#             print(root.number_visits)
            all_leafs.append(leaf._id)
            board = tensorize(leaf.state)
            legal_moves = leaf.legal_moves
            leafs.append(leaf)
            boards.append(board)
            legal_moves_sets.append(legal_moves)
        unique_leafs = set([leaf._id for leaf in leafs])
#         print(f"We've selected the following leafs: {unique_leafs}")
#         assert len(unique_leafs) == current_batch_size, f'you have selected {len(unique_leafs)} unique leafs instead of {current_batch_size}' 
        with torch.no_grad():
            child_priors, value_estimates = network(torch.stack(boards), legal_moves_sets)
            child_priors = F.softmax(child_priors, dim=-1)
        for leaf, child_prior, value_estimate in zip(leafs, child_priors, value_estimates):
            leaf.backup(value_estimate)
            if not leaf.terminal:
                leaf.expand(child_prior)
        num_reads -= min(batch_size, num_reads)
#     print(f'In total, we selected {len(set(all_leafs))} leafs')
    return root


#     return max(root.children.items(),
#        key=lambda item: item[1].number_visits)


class DummyNode(object):
    """A fake node of a MCTS search tree.
    This node is intended to be a placeholder for the root node, which would
    otherwise have no parent node. If all nodes have parents, code becomes
    simpler."""

    def __init__(self):
        self.parent = None
        self.child_number_visits = collections.defaultdict(float)
        self.child_priors = collections.defaultdict(float)
        self.child_total_value = collections.defaultdict(float)
        self.child_terminals = np.zeros([1], dtype=np.bool)
        self.child_rewards = np.zeros([1], dtype=np.float)

class UCTNode():
    def __init__(self, state,
                 move, parent=None, c_puct=None):
        self.state = state
        self.move = move
        self.is_expanded = False
        self.parent = parent  # Optional[UCTNode]
        self.moves = [] # List[int]
        self.children = []  # List[UCTNode]
        self._id = ''.join([random.choice('abcdefghijklmnopqrtuvwxyz') for _ in range(10)])
        if state == None:
            self.legal_moves = None
        else:
            self.legal_moves = all_legal_moves(state.board, state.ko)
        self.child_priors = np.zeros(
            82, dtype=np.float32)
        self.child_total_value = np.zeros(
            82, dtype=np.float32)
        self.child_number_visits = np.zeros(
            82, dtype=np.float32)
        self.child_terminals = np.zeros(82, dtype=np.bool)
        self.child_rewards = np.zeros(82, dtype=np.float)
        
    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]
    
    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value
        
    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]
    
    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value
    
    @property
    def terminal(self):
        return self.parent.child_terminals[self.move]
    
    @terminal.setter
    def terminal(self, value):
        self.parent.child_terminals[self.move] = value
    
    @property
    def reward(self):
        return self.parent.child_rewards[self.move]
        
    @reward.setter
    def reward(self, value):
        self.parent.child_rewards[self.move] = value
        
    def child_Q(self):
        return self.child_total_value * self.state.to_play / (1 + self.child_number_visits)
    
    def child_U(self):
        return 2 * self.child_priors.detach() * (math.sqrt(self.number_visits)
             / (1 + self.child_number_visits))
    
    def best_child(self):
        node_values = (self.child_U() + self.child_Q()) * \
                                      self.state.to_play
        best_child_index = np.argmax(node_values[self.legal_moves])
        return self.children[self.legal_moves[best_child_index]]
    
    def select_leaf(self):
        current = self
        while current.is_expanded:
            current.number_visits += 1
            current.total_value -= current.state.to_play
            current = current.best_child()
        else:
            current.materialize()
            current.number_visits += 1
            current.total_value -= current.state.to_play
        return current
    
    def expand(self, child_priors):
        if self.is_expanded: return
        self.is_expanded = True
        self.child_priors = child_priors
        for move in range(82):
            self.add_child(move)
            
    def materialize(self):
        # nodes are initialized with empty states; only when expanding, do we call the transition function
        if not self.state:
            self.state = transition_function(self.parent.state, self.move)
            self.legal_moves = all_legal_moves(self.state.board, self.state.ko)
            self.terminal = is_terminal(self.state)
#             print(f'node {self._id} is {self.terminal} terminal')
            self.reward = reward_function(self.state)

    def add_child(self, move):
        self.moves.append(move)
        if move in self.legal_moves:
            new_node = UCTNode(
                None, move, parent=self)
            self.children.append(new_node)
        else:
            self.child_terminals[move] = True
            self.child_rewards[move] = 0
            self.children.append(None)
        
    def backup(self, value_estimate):
        current = self
        while current.parent is not None:
            if current.terminal:
#                 print(f'terminal node. current value: {current.total_value}, reward: {self.reward}, to_play: {self.state.to_play}') 
                current.total_value += current.reward + current.state.to_play
                value_estimate = torch.as_tensor([current.reward])
#                 print(f'aaand the new value is {current.total_value}') 
            else:
                current.total_value += value_estimate + current.state.to_play
            current = current.parent