import random
from typing import Dict, List, Tuple
from functools import partial
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
import tqdm

TEAM_NAME = "Deep Learners"

def choose_move_no_value_fn(board: List[str]) -> Tuple[int, str]:
    return choose_move(board, {})

all_possible_moves = [(cell, mark) for cell in range(9) for mark in 'XO']

def make_move(board, move):
    return board[:move[0]] + move[1] + board[move[0]+1:]

# def choose_move(board: List[str], value_function: Dict, epsilon=0.05) -> Tuple[int, str]:
#     board = ''.join(board)
#     action_values = value_function.get(board, {})
#     possible_moves = [m for m in all_possible_moves if board[m[0]] == ' ']
#     if random.random() > epsilon:
#       best_move = sorted(possible_moves, key=lambda m: value_function.get(make_move(board, m), 0),reverse=True)[0]
#       return best_move
#     else:
#       return random.choice(possible_moves)

def choose_move(board: List[str], value_function: Dict, epsilon=0.05) -> Tuple[int, str]:
    board = ''.join(board)
    action_values = value_function.get(board, {})
    possible_moves = [m for m in all_possible_moves if board[m[0]] == ' ']
    if random.random() > epsilon:
        move_values = [value_function.get(make_move(board, m), 0) for m in possible_moves]
        best_move_value = max(move_values)
        best_moves = [m for (m, v) in zip(possible_moves, move_values) if abs(v - best_move_value) < 0.001]
        best_move = random.choice(best_moves)
        return best_move
    else:
        return random.choice(possible_moves)



def train(n_episodes = 100_000, gamma = 0.99, alpha = 0.95) -> Dict:
    """Write this function to train your algorithm.

    Returns:
         Value function dictionary used by your agent. You can
         structure this how you like, however your choose_move must
         be able to use it.
    """
    env = WildTictactoeEnv(choose_move_no_value_fn)
    value_fn = {}
    for episode in tqdm.tqdm(range(n_episodes)):
        state, reward, done, info = env.reset(0)
        while not done:
            old_state = ''.join(state)
            old_reward = reward
            move = choose_move(state, value_fn)
            opponent_state = make_move(old_state, move)
            state, reward, done, info = env.step(move, 0)
            state = ''.join(state)
            old_evaluation = value_fn.get(old_state,0)
            new_evaluation = value_fn.get(state,0)
            value_fn[old_state] = old_evaluation*(1-alpha) + alpha*(old_reward + gamma*new_evaluation)
        value_fn[state] = abs(reward)
    return value_fn




if __name__ == "__main__":

    ## Example workflow, feel free to edit this! ###
    try:
        my_value_fn = load_dictionary(TEAM_NAME)
    except:
        print("warn - no dict found")
    my_value_fn = train()
    save_dictionary(my_value_fn, TEAM_NAME)
    my_value_fn = load_dictionary(TEAM_NAME)

    def choose_move_no_value_fn(board: List[str]) -> Tuple[int, str]:
        """
        The arguments in play_wild_ttt_game() require functions 
         that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(board, my_value_fn)

    # Code below plays a single game of Wild Tic-Tac-Toe vs a random
    # opponent, think about how you might want to adapt this to
    # test the performance of your algorithm.
    total_return = 0
    for i in range(100):
        total_return += play_wild_ttt_game(
            your_choose_move=partial(choose_move, value_function=my_value_fn),
            opponent_choose_move=choose_move_randomly,
            game_speed_multiplier=1000000,
            verbose=False,
        )
    print(total_return)
    # Below renders a game graphically. You must click to take turns
    #render(choose_move)

    check_submission()
