import random
from typing import Tuple, Optional, Dict, List, Literal, cast
import dataclasses
from tqdm import tqdm

import numpy as np

Action = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]
StateID = Tuple[Tuple[int], int]


@dataclasses.dataclass
class State:
    board: List[int]
    player_to_move: Literal[-1, 1]

    @property
    def key(self) -> StateID:
        return tuple(self.board), self.player_to_move

    @property
    def other_player(self) -> Literal[-1, 1]:
        return cast(Literal[-1, 1], -self.player_to_move)


def get_possible_actions(state: State) -> List[Action]:
    return [cast(Action, index) for index, piece in enumerate(state.board) if piece == 0]


def choose_move_randomly(state: State) -> Action:
    poss_moves = get_possible_actions(state)
    return poss_moves[int(random.random() * len(poss_moves))]


def transition_function(state: State, action: Action) -> State:
    assert state.board[action] == 0, "You moved onto a square that already has a counter on it!"
    board = state.board.copy()
    board[action] = state.player_to_move
    return State(board, state.other_player)


def reward_function(successor_state: State) -> int:
    winner = _check_winner(successor_state)
    return winner if winner is not None else 0


def is_terminal(previous_action: Action, successor_state: State) -> bool:
    return _check_winner_given_last_move(previous_action, successor_state) is not None or _is_draw(
        successor_state
    )


def check_winning_set(pieces: List[int]) -> bool:
    unique_pieces = set(pieces)
    return 0 not in unique_pieces and len(unique_pieces) == 1


def _is_draw(state: State) -> bool:
    return all(c != 0 for c in state.board)


def _check_winner_given_last_move(previous_action: Action, state: State) -> Optional[int]:
    piece_played = state.board[previous_action]
    if state.board.count(piece_played) < 3:
        return

    # Check row
    if check_winning_set([state.board[i + 3 * (previous_action // 3)] for i in range(3)]):
        return piece_played

    # Check col
    if check_winning_set([state.board[i + previous_action % 3] for i in range(0, 9, 3)]):
        return piece_played

    # If in diagonals
    if previous_action % 2 == 0:
        if previous_action in {0, 4, 8} and check_winning_set([state.board[i] for i in [0, 4, 8]]):
            return state.board[0]

        if previous_action in {2, 4, 6} and check_winning_set([state.board[i] for i in [2, 4, 6]]):
            return state.board[2]


def _check_winner(state: State) -> Optional[int]:
    board = [[state.board[e], state.board[e + 1], state.board[e + 2]] for e in range(0, 9, 3)]
    # Check rows
    for row in board:
        if check_winning_set(row):
            return row[0]

    # Check columns
    for column in [*zip(*board)]:
        if check_winning_set(column):
            return column[0]

    # Check major diagonal
    size = 3
    major_diagonal = [board[i][i] for i in range(size)]
    if check_winning_set(major_diagonal):
        return major_diagonal[0]

    # Check minor diagonal
    minor_diagonal = [board[i][size - i - 1] for i in range(size)]
    return minor_diagonal[0] if check_winning_set(minor_diagonal) else None


class TictactoeMechanics:
    def __init__(self):
        self.piece = 1
        self.done = False
        self.state = State(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            random.choice([1, -1]),
        )

    def mark_square(self, piece: int):
        assert (
            self.state.board[piece] == 0
        ), "You moved onto a square that already has a counter on it!"
        self.state.prev_move = piece
        self.state.board[piece] = self.state.player_to_move

    def __repr__(self):
        return str(np.array(self.state.board).reshape((3, 3))) + "\n"

    def is_board_full(self):
        """Check if the board is full by checking for empty cells after flattening board."""
        return all(c != 0 for c in self.state.board)

    def switch_player(self) -> None:
        self.state.player_to_move = 1 if self.state.player_to_move == -1 else -1

    def _step(self, action: Action, verbose: bool = False) -> Tuple[float, Dict]:
        assert not self.done, "Game is done. Call reset() before taking further steps."
        self.mark_square(action)

        winner = _check_winner_given_last_move(action, self.state)
        reward = 1.0 if winner else 0.0
        self.done = winner is not None or self.is_board_full()
        if verbose:
            print(self)
            if winner is not None:
                print(f"{self.state.player_to_move} wins!")
            elif self.is_board_full():
                print("Game Drawn")

        self.switch_player()
        info = {"winner": winner}

        return reward, info

    def step(
        self, action: Action, verbose: bool = False
    ) -> Tuple[State, Optional[float], bool, Dict]:
        # Take your move
        reward, info = self._step(action, verbose)

        # Take opponent move
        if not self.done:
            opponent_action = choose_move_randomly(self.state)
            opponent_reward, info = self._step(opponent_action, verbose)
            reward -= opponent_reward

        return self.state, reward, self.done, info

    def reset(self) -> Tuple[State, Optional[float], bool, Dict]:
        self.state = State(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            random.choice([1, -1]),
        )
        self.done = False
        reward = 0
        info = {}

        if self.state.player_to_move != self.piece:
            opponent_action = choose_move_randomly(self.state)
            reward, info = self._step(opponent_action)
        return self.state, reward, self.done, info


def validate_mcts(mcts_class, verbose: bool = False):
    env = TictactoeMechanics()

    n_wins, n_losses, n_draws = 0, 0, 0
    for _ in tqdm(range(100)):
        # Setup everything for a new episode
        state, reward, done, _ = env.reset()
        mcts = mcts_class(
            initial_state=state,
            rollout_policy=lambda x: get_possible_actions(x)[
                int(random.random() * len(get_possible_actions(x)))
            ],
            explore_coeff=0.5,
            verbose=verbose,
        )

        # Run episode loop
        while not done:
            for _ in range(400):
                mcts.do_rollout()

            action = mcts.choose_action()

            state, reward, done, _ = env.step(action, verbose=False)

            mcts.prune_tree(action, state)

        # Count up wins/losses
        if reward == 1:
            n_wins += 1
        elif reward == -1:
            n_losses += 1
        else:
            n_draws += 1

        print("wins:", n_wins, "losses:", n_losses, "draws:", n_draws)
