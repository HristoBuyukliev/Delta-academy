import time
from pathlib import Path
from typing import Callable, Dict

import delta_utils.check_submission as checker
from game_mechanics import TronEnv
from torch import nn


def check_submission(choose_move: Callable, mcts) -> None:

    example_state, _, _, _ = TronEnv(opponent_choose_move=lambda x: x).reset()
    expected_choose_move_return_type = int
    expected_pkl_output_type = None
    t1 = time.time()
    choose_move(example_state, mcts)
    t2 = time.time()
    assert t2 - t1 < 0.5, "Oh no your choose_move takes longer than 500ms, this is too slow!"

    return checker.check_submission(
        example_state=example_state,
        expected_choose_move_return_type=expected_choose_move_return_type,
        expected_pkl_type=expected_pkl_output_type,
        pkl_file=None,
        pkl_checker_function=None,
        current_folder=Path(__file__).parent.resolve(),
        choose_move_extra_argument={"mcts": mcts},
    )
