from typing import Dict, Optional, Tuple

from env import State, get_possible_actions, Action, is_terminal

NodeID = Tuple[Tuple[Tuple[int], int], int]


class Node:
    def __init__(self, state: State, last_action: Optional[Action]):
        self.state = state
        self.last_action = last_action
        self.is_terminal = is_terminal(last_action, state) if last_action is not None else False
        # No guarantee that these NODES exist in the MCTS TREE!
        self.child_states = self._get_possible_children()
        self.key: NodeID = self.state.key, last_action

    def _get_possible_children(self) -> Dict[Action, State]:
        """Gets the possible children of this node."""
        if self.is_terminal:
            return {}
        children = {}
        for action in get_possible_actions(self.state):
            state = self.state.board.copy()
            state[action] = self.state.player_to_move
            children[action] = State(state, self.state.other_player)
        return children
