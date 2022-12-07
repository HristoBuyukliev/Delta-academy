from check_submission import check_submission
from game_mechanics import (
    State,
    TronEnv,
    choose_move_randomly,
    choose_move_square,
    human_player,
    is_terminal,
    play_tron,
    reward_function,
    rules_rollout,
    transition_function,
)

TEAM_NAME = "Hristo"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


class MCTS:
    def __init__(self):
        """Implement your MCTS algorithm here.

        This class' state persists across calls to choose_move!
        """
        pass

    def prune(self):
        pass

    def set_initial_state(self, state: State):
        pass


def choose_move(state: State, mcts: MCTS) -> int:
    """Called during competitive play. It acts greedily given current state of the game. It returns
    a single action to take.

    Args:
        state: a State object containing the positions of yours and your opponents snakes

    Returns:
        The action to take
    """
    raise NotImplementedError("You need to implement this function!")


if __name__ == "__main__":

    # Example workflow, feel free to edit this!

    # Make sure this passes, or your solution will not work in the tournament!!

    my_mcts = MCTS()
    check_submission(choose_move, my_mcts)

    def choose_move_no_network(state: State) -> int:
        """The arguments in play_game() require functions that only take the state as input.

        This converts choose_move() to that format while persisting your mcts.
        """
        return choose_move(state, mcts=my_mcts)

    # Play against your bot!
    play_tron(
        your_choose_move=human_player,
        opponent_choose_move=choose_move_no_network,
        game_speed_multiplier=3,
        render=True,
        verbose=False,
    )
