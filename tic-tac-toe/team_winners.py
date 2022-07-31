from collections import defaultdict
import math
import random
from typing import Dict, List, Tuple, Set
import os
import numpy as np
from check_submission import check_submission
from game_mechanics import (
    Cell,
    WildTictactoeEnv,
    choose_move_randomly,
    load_dictionary,
    play_wild_ttt_game,
    render,
    save_dictionary,
)

# Notes:
# board = List[str]
# action = Tuple[int, str]

TEAM_NAME = "Delta Winners"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"
team_name = TEAM_NAME


def _matr_to_tup(matr: np.array) -> Tuple[str]:
    return tuple([str(x) for xs in matr for x in xs])


def get_symmetric_layouts(board: Tuple[str]) -> Set[Tuple[str]]:
    """
    Get all the layouts that are basically the same as the current one.
    That means rotations and reflections (in space and O-X)
    Args:
        board:

    Returns:
        Set of modified boards
    """
    output = list(board[::-1])
    dd_board = np.reshape(board, (3, 3))
    reflection = board[:]
    reflection = ["I" if item == "O" else item for item in reflection]
    reflection = ["O" if item == "X" else item for item in reflection]
    reflection = ["X" if item == "I" else item for item in reflection]
    board_mirror_r = list(reflection[::-1])
    dd_board_mirror_r = np.reshape(board_mirror_r, (3, 3))
    dd_board_ref = np.reshape(reflection, (3, 3))
    dd_board_mirror = np.reshape(output, (3, 3))
    init_boards = [dd_board, dd_board_ref, dd_board_mirror, dd_board_mirror_r]
    
    rots = 3
    all_boards = set()
    
    for _board in init_boards:
        for kk in range(0, rots):
            cur_rot = np.rot90(_board, k=kk)
            all_boards.add(_matr_to_tup(cur_rot))
            all_boards.add(_matr_to_tup(np.fliplr(cur_rot)))
            all_boards.add(_matr_to_tup(np.flipud(cur_rot)))
    
    return all_boards
    

def possible_boards(board: Tuple[str], value: float) -> Dict[Tuple[str], float]:
    dd_board = np.reshape(board, (3, 3))
    reflection = board
    reflection = ["I" if item == "O" else item for item in reflection]
    reflection = ["O" if item == "X" else item for item in reflection]
    reflection = ["X" if item == "I" else item for item in reflection]
    dd_board_ref = np.reshape(reflection, (3, 3))
    r1 = np.rot90(dd_board)
    r2 = np.rot90(r1)
    r3 = np.rot90(r2)
    r4 = np.rot90(dd_board_ref)
    r5 = np.rot90(r4)
    r6 = np.rot90(r5)
    board1 = tuple([x for xs in r1 for x in xs])
    board2 = tuple([x for xs in r2 for x in xs])
    board3 = tuple([x for xs in r3 for x in xs])
    board4 = tuple([x for xs in r4 for x in xs])
    board5 = tuple([x for xs in r5 for x in xs])
    board6 = tuple([x for xs in r6 for x in xs])
    boards = {
        tuple(board): value,
        board1: value,
        board2: value,
        board3: value,
        tuple(reflection): value,
        board4: value,
        board5: value,
        board6: value,
    }
    return boards


def get_possible_actions(board: Tuple[str]) -> List[Tuple[int, str]]:
    empty_spots = [count for count, item in enumerate(board) if item == Cell.EMPTY]
    counters = [Cell.O, Cell.X]
    possible_actions = [[pos, counter] for pos in empty_spots for counter in counters]
    return possible_actions


def transition_function(board: Tuple[str], action: Tuple[int, str]) -> List[str]:
    new_board = list(board)
    position, counter = action
    assert board[position] == Cell.EMPTY
    new_board[position] = counter
    new_board = tuple(new_board)
    return new_board


def choose_greedy_action(current_state: Tuple[str], value_fn: Dict) -> Tuple[int, str]:
    """
    Choose the greedy action, given the value function, possible actions and current state.

    The 'greedy action' is the action that will lead to the highest value state according
        to your current value function.

    Args:
         current_state: tuple representing the current state

         value_fn: numpy array representing the value function. Can get the value of
                    a state with `value_fn[state]`.

    Returns:
        The action to take which maximises the value of the successor state.
    """

    max_value = -float("inf")

    possible_actions = get_possible_actions(current_state)
    best_actions = []
    for poss_action in possible_actions:
        poss_new_state = transition_function(current_state, poss_action)
        if value_fn[poss_new_state] > max_value:
            best_actions = [poss_action]
            max_value = value_fn[poss_new_state]
        elif math.isclose(value_fn[poss_new_state], max_value, abs_tol=1e-4):
            best_actions.append(poss_action)
    return random.choice(best_actions)


def choose_move(
    board: Tuple[str], value_function: Dict[Tuple[str], float], eps=0.0
) -> Tuple[int, str]:
    """
    This is what will be called during competitive play.
    It takes the current state of the board as input.
    It returns a single move to play.

    Args:
        board: list representing the board.
                (see README Technical Details for more info)

        value_function: The dictionary output by train().

    Returns:
        position (int): The position to place your piece
                        (an integer 0 -> 8), where 0 is
                        top left and 8 is bottom right.
        counter (str): The counter to place. "X" or "O".

    It's important that you think about exactly what this
     function does when you submit, as it will be called
     in order to take your turn!
    """

    if random.random() < eps:
        # Chooses a random position on the board and places a random counter there.
        possible_actions = get_possible_actions(board)
        action = random.choice(possible_actions)
    else:
        # Pick the best move
        action = choose_greedy_action(board, value_function)

    action = tuple(action)
    return action


def average_symmetric_boards(value_fn: Dict[Tuple[str], float]):
    """
    Average the values of symmetric boards
    Args:
        value_fn:

    Returns:

    """
    known_boards = set(value_fn.keys())
    counted_boards = set()
    while len(known_boards) > 0:
        base_board = known_boards.pop()
        if base_board in counted_boards:
            continue
        symmetries = get_symmetric_layouts(base_board)
        counted_boards.update(symmetries)
        known_boards.difference_update(symmetries)
        values = [value_fn[sym] for sym in symmetries if sym in value_fn]
        num_vals = len(values)
        if num_vals > 0:
            mean_val = sum(values) / num_vals
            for sym in symmetries:
                value_fn[sym] = mean_val
    

def train() -> Dict[Tuple[str], float]:
    """Write this function to train your algorithm.

    Returns:
         Value function dictionary used by your agent. You can
         structure this how you like, however your choose_move must
         be able to use it.
    """
    # TODO What's a good alpha value?
    init_alpha = 0.5
    init_train_eps = 0.2

    # Training for 100k episodes take ~30 seconds on my machine
    num_episodes = int(1e6)
    # The game is pretty short, future is roughly the same as the present
    gamma = 1

    # random.seed(123984)

    # value_function = Dict[Tuple, float]
    # value_function maps board -> expected reward (board needs to be a tuple to hashable)
    # The expected reward is to the player who just made a move
    value_fn = defaultdict(float)

    opponent_random = choose_move_randomly
    # 'Self' opponent will stay a little unpredictable
    opponent_self = lambda board: choose_move(board, value_fn, eps=init_train_eps)

    opponent_choose_move = opponent_self
    ttt_env = WildTictactoeEnv(opponent_choose_move)

    for ep in range(num_episodes):
        # Use a higher alpha factor in the beginning
        if ep < 1000:
            alpha = init_alpha
            train_eps = init_train_eps
        else:
            alpha = init_alpha / 10.0
            train_eps = init_train_eps / 10.0

        cur_state, cur_reward, done, info = ttt_env.reset()
        cur_state = tuple(cur_state)

        while not done:
            action = choose_move(cur_state, value_fn, train_eps)

            # If the game is still going, `next_state` will include a move from the opponent
            next_state, cur_reward, done, info = ttt_env.step(action)
            next_state = tuple(next_state)

            if done:
                # The game is now over. We store value of the final state
                # Take the abs because we use the number to choose what we move into, with our move.
                # So even if the opponent won we could get there next time.
                value_fn[next_state] = abs(cur_reward)

                # Two possibilities:
                # 1. From `cur_state`, we took a move and won. That means we never want to move into it,
                # because then our opponent will take a move and win.

                # 2. From `cur_state`, we took a move, our opponent took a move, and then we lost or drew.
                # That means we might want to move into the state, because then our opponent will move and we can
                # win or draw. So we negate the reward.
                value_fn[cur_state] = -cur_reward
            else:
                # When we don't hit a final game state, we do the normal TD update rule.
                tmp1 = (1 - alpha) * value_fn[cur_state]
                tmp2 = alpha * (cur_reward + gamma * value_fn[next_state])

                value_fn[cur_state] = tmp1 + tmp2

            cur_state = next_state
    
        if False and ep % 10000 == 0 or ep == num_episodes - 1:
            print(f"Episode {ep}: Value_fn size {len(value_fn)}")

    average_symmetric_boards(value_fn)

    # print(f"Final: Value_fn size {len(value_fn)}")

    return value_fn


if True and __name__ == "__main__":
    from game_mechanics import HERE
    from pprint import pprint

    dict_path = os.path.join(HERE, f"dict_{TEAM_NAME}.pkl")
    force_retrain = False
    need_train = force_retrain or not os.path.exists(dict_path)

    if need_train:
        my_value_fn = train()
        save_dictionary(my_value_fn, TEAM_NAME)

    my_value_fn = load_dictionary(TEAM_NAME)

    def choose_move_no_value_fn(board: List[str]) -> Tuple[int, str]:
        """
        The arguments in play_wild_ttt_game() require functions
         that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(board, my_value_fn, eps=0.0)

    # Code below plays a single game of Wild Tic-Tac-Toe vs a random
    # opponent, think about how you might want to adapt this to
    # test the performance of your algorithm.

    # Random opponent
    opponent_choose_move = choose_move_randomly

    # Self
    # opponent_choose_move = choose_move_no_value_fn

    if False:

        # random.seed(383)

        num_games = 1000

        total_return = 0
        results = defaultdict(int)

        for gg in range(num_games):
            cur_return = play_wild_ttt_game(
                your_choose_move=choose_move_no_value_fn,
                opponent_choose_move=opponent_choose_move,
                game_speed_multiplier=1e9,
                verbose=False,
            )

            total_return += cur_return

            results[cur_return] += 1

        labels = {+1: "Win", 0: "Draw", -1: "Loss"}

        disp_dict = {labels[x]: val for x, val in results.items()}

        pprint(disp_dict, indent=2)
        print(f"Total return: {total_return}")

    # Below renders a game graphically. You must click to take turns
    if False:
        # random.seed()
        render(choose_move_no_value_fn, opponent_choose_move)

    check_submission()
