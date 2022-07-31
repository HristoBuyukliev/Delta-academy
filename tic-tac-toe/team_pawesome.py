import random
from typing import Dict, List, Tuple

from check_submission import check_submission
from game_mechanics import (
    Cell,
    WildTictactoeEnv,
    choose_move_randomly,
    load_dictionary,
    play_wild_ttt_game,
    render,
    save_dictionary,
    convert_to_indices,
    flatten_board,
)
import numpy as np
import math
import copy

TEAM_NAME = "pawesome"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", \
    "Please change your TEAM_NAME!"

alpha = 0.2
gamma = 0.95
epsilon = 0

def stringify_board(board: List[str]) -> str:
  return '#'.join(board)

def flip_board(board: List[str]) -> List[str]:
  for i, s in enumerate(board):
    if s == Cell.X:
      board[i] = Cell.O
    if s == Cell.O:
      board[i] = Cell.X
  return board   

def get_possible_actions(board: List[str]):
  possible_positions = [count for count, item in enumerate(board) if item == Cell.EMPTY]
  possible_counters = [Cell.O, Cell.X]
  return [(pos, count) for pos in possible_positions for count in possible_counters]

def train(opponent_choose_move, value_fn) -> Dict:
    """Write this function to train your algorithm.

    Returns:
         Value function dictionary used by your agent. You can
         structure this how you like, however your choose_move must
         be able to use it.
    """
    game = WildTictactoeEnv(opponent_choose_move)
    for i in range(100000):
      state, reward, done, info = game.reset(False)
      before_before_state = state
      before_state = state
      current_state = state

      while not done:
        rand_num = random.random()
        possible_actions = get_possible_actions(current_state)
        if rand_num < epsilon:
          action = random.choice(possible_actions)
        else:
          action = choose_move(current_state, value_fn)
        pos, counter = action 
        new_state_for_opponent = copy.deepcopy(current_state)
        new_state_for_opponent[pos] = counter

        state, reward, done, info = game.step(action, False)
        
        # get state strings
        current_state_str = stringify_board(current_state)
        new_state_for_opponent_str = stringify_board(new_state_for_opponent)
        state_str = stringify_board(state)
        if current_state_str not in value_fn:
          value_fn[current_state_str] = 0
        if new_state_for_opponent_str not in value_fn:
          value_fn[new_state_for_opponent_str] = 0
        if state_str not in value_fn:
          value_fn[state_str] = 0

        # update value function
        value_fn[new_state_for_opponent_str] = (1 - alpha) * value_fn[new_state_for_opponent_str] + alpha * (reward + gamma * value_fn[state_str])
        value_fn[current_state_str] = (1 - alpha) * value_fn[current_state_str] + alpha * (-reward + gamma * value_fn[state_str])
        
        new_state_for_opponent_flip_str = stringify_board(flip_board(new_state_for_opponent))
        if new_state_for_opponent_flip_str not in value_fn:
          value_fn[new_state_for_opponent_flip_str] = 0
        value_fn[new_state_for_opponent_str] = (value_fn[new_state_for_opponent_str] + value_fn[new_state_for_opponent_flip_str]) / 2
        value_fn[new_state_for_opponent_flip_str] = value_fn[new_state_for_opponent_str]
        
        current_state_flip_str = stringify_board(flip_board(current_state))
        if current_state_flip_str not in value_fn:
          value_fn[current_state_flip_str] = 0
        value_fn[current_state_str] = (value_fn[current_state_str] + value_fn[current_state_flip_str]) / 2
        value_fn[current_state_flip_str] = value_fn[current_state_str]

        before_before_state = current_state
        before_state = new_state_for_opponent
        current_state = state

      # update winning condition in value function
      if reward == 1:
        value_fn[stringify_board(current_state)] = 1
        value_fn[stringify_board(flip_board(current_state))] = 1
        value_fn[stringify_board(before_before_state)] = -1
        value_fn[stringify_board(flip_board(before_before_state))] = -1
      if reward == -1:
        value_fn[stringify_board(current_state)] = 1
        value_fn[stringify_board(flip_board(current_state))] = 1
        value_fn[stringify_board(before_state)] = -1
        value_fn[stringify_board(flip_board(before_state))] = -1
    
    return value_fn


def choose_move(board: List[str], value_function: Dict) -> Tuple[int, str]:
    """
    TODO: WRITE THIS FUNCTION

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
    max_value = -np.Inf
    best_actions = []
    possible_actions = get_possible_actions(board)
    for poss_position, possible_counter in possible_actions:
      board_copy = copy.deepcopy(board)
      board_copy[poss_position] = possible_counter
      possible_new_state = stringify_board(board_copy)
      if possible_new_state not in value_function:
        value_function[possible_new_state] = 0
      if value_function[possible_new_state] > max_value:
          best_actions = [(poss_position, possible_counter)]
          max_value = value_function[possible_new_state]
      elif math.isclose(value_function[possible_new_state], max_value, abs_tol=1e-6):
          best_actions.append((poss_position, possible_counter))

    return random.choice(best_actions)
    


if __name__ == "__main__":

    ## Example workflow, feel free to edit this! ###
    # opponent_choose_move = choose_move_randomly
    # my_value_fn = train(opponent_choose_move, {})
    # save_dictionary(my_value_fn, TEAM_NAME)
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
  
    count_lose = 0
    draw = 0
    for i in range(10000):
      total_return = play_wild_ttt_game(
          your_choose_move=choose_move_no_value_fn,
          opponent_choose_move=choose_move_randomly,
          game_speed_multiplier=1000000,
          verbose=False,
      )
      if total_return < 0:
        count_lose += 1
      if total_return == 0:
        draw += 1
    print(count_lose)
    print(draw)

    # Below renders a game graphically. You must click to take turns
    # render(choose_move_no_value_fn)

    # check_submission()
