from typing import Any, Dict

import numpy as np

from tqdm import tqdm
# from check_submission import check_submission
from game_mechanics import GoEnv, choose_move_randomly, load_pkl, play_go, save_pkl

from torch import nn

TEAM_NAME = "Team Henry"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


def train():
    raise NotImplementedError()


def choose_move(observation: np.ndarray, legal_moves: np.ndarray, neural_network: nn.Module) -> int:
    """Called during competitive play. It acts greedily given current state of the board and value
    function dictionary. It returns a single move to play.

    Args:
        state:

    Returns:
    """
    return choose_move_randomly(observation, legal_moves)


if __name__ == "__main__":

    ## Example workflow, feel free to edit this! ###
    file = train()
    save_pkl(file, TEAM_NAME)

    # my_network = load_pkl(TEAM_NAME)

    # Code below plays a single game against a random
    #  opponent, think about how you might want to adapt this to
    #  test the performance of your algorithm.
    def choose_move_no_network(observation: np.ndarray, legal_moves: np.ndarray) -> int:
        """The arguments in play_game() require functions that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(observation, legal_moves, None)
        # return len(observation) ** 2

    # check_submission(
    #     TEAM_NAME, choose_move_no_network
    # )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    play_go(
        your_choose_move=choose_move_no_network,
        opponent_choose_move=choose_move_no_network,
        game_speed_multiplier=1,
        render=False,
        verbose=True,
    )
