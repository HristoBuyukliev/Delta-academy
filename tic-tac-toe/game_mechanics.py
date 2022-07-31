import os
import pickle
import random
from pathlib import Path
from time import sleep
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pygame

HERE = Path(__file__).parent.resolve()

######## Below are classes and functions you will use to implement 
# your wild-tic-tac-toe AI ######


########## USEFUL FUNCTIONS ##########


def save_dictionary(my_dict: Dict, team_name: str) -> None:
    assert isinstance(
        my_dict, dict
    ), f"train() function should output a dict, but got: {type(my_dict)}"
    assert "/" not in team_name, "Invalid TEAM_NAME. '/' are illegal in TEAM_NAME"

    n_retries = 5
    dict_path = os.path.join(HERE, f"dict_{team_name}.pkl")
    for attempt in range(n_retries):
        try:
            with open(dict_path, "wb") as f:
                pickle.dump(my_dict, f)
            load_dictionary(team_name)
            return
        except Exception as e:
            if attempt == n_retries - 1:
                raise


def load_dictionary(team_name: str) -> Dict:
    dict_path = os.path.join(HERE, f"dict_{team_name}.pkl")
    with open(dict_path, "rb") as f:
        return pickle.load(f)


def choose_move_randomly(board: List[str]) -> Tuple[int, str]:
    position: int = random.choice([count for count, item in enumerate(board) if item == Cell.EMPTY])
    counter: str = random.choice([Cell.O, Cell.X])
    return position, counter


def play_wild_ttt_game(
    your_choose_move: Callable[[List[str]], Tuple[int, str]],
    opponent_choose_move: Callable[[List[str]], Tuple[int, str]],
    game_speed_multiplier: float = 1.0,
    verbose: bool = False,
) -> int:
    """Play a game where moves are chosen by `your_choose_move()` and 
     `opponent_choose_move()`. Who goes first is chosen at random. You
     can render the game by setting `render=True`.

    Args:
        your_choose_move: function that chooses move (takes board as input)
        opponent_choose_move: function that picks your opponent's next move
        game_speed_multiplier: multiplies the speed of the game. High == fast
        verbose: whether to print board states to console. For debugging

    Returns: total_return, which is the sum of return from the game
    """
    total_return = 0
    game = WildTictactoeEnv(opponent_choose_move)
    state, reward, done, info = game.reset(verbose)

    sleep(1 / game_speed_multiplier)

    while not done:
        action = your_choose_move(state)
        state, reward, done, info = game.step(action, verbose)

        total_return += reward
        sleep(1 / game_speed_multiplier)

    return total_return


class Cell:
    """You will need to interact with this!

    This class represents the state of a single square of the tic-tac-toe 
     board. An X counter is represented by Cell.X An O counter is 
     represented by Cell.O A blank square represented by Cell.EMPTY
    """

    EMPTY = " "
    X = "X"
    O = "O"


########## LESS USEFUL ##########


class Player:
    """This class defines which players turn it is.

    If player: It is the turn of the player passing action directly to step.
     If opponent: It is the turn of the opponent_choose_move function passed
     to WildTictactoeEnv's __init__()
    """

    player = "player"
    opponent = "opponent"


def mark_square(board: List[List[str]], row: int, col: int, counter: str) -> List[List[str]]:
    board[row][col] = counter
    return board


def is_board_full(board: List[List[str]]) -> bool:
    """Check if the board is full by checking for empty cells after flattening board."""
    return all(c != Cell.EMPTY for c in [i for sublist in board for i in sublist])


def _check_winning_set(iterable: Iterable[str]) -> bool:
    unique_pieces = set(iterable)
    return Cell.EMPTY not in unique_pieces and len(unique_pieces) == 1


def is_winner(board: List[List[str]]) -> bool:
    # Check rows
    for row in board:
        if _check_winning_set(row):
            return True

    # Check columns
    for column in [*zip(*board)]:
        if _check_winning_set(column):
            return True

    # Check major diagonal
    size = len(board)
    major_diagonal = [board[i][i] for i in range(size)]
    if _check_winning_set(major_diagonal):
        return True

    # Check minor diagonal
    minor_diagonal = [board[i][size - i - 1] for i in range(size)]
    if _check_winning_set(minor_diagonal):
        return True

    return False


def get_empty_board() -> List[List[str]]:
    return [
        [Cell.EMPTY, Cell.EMPTY, Cell.EMPTY],
        [Cell.EMPTY, Cell.EMPTY, Cell.EMPTY],
        [Cell.EMPTY, Cell.EMPTY, Cell.EMPTY],
    ]


class WildTictactoeEnv:
    def __init__(
        self,
        opponent_choose_move: Callable[[List], Tuple[int, str]] = choose_move_randomly,
    ):
        self.opponent_choose_move = opponent_choose_move
        self.done: bool = False
        self.board = get_empty_board()

    def __repr__(self) -> str:
        return str(np.array([x for xs in self.board for x in xs]).reshape((3, 3))) + "\n"

    def switch_player(self) -> None:
        self.player_move: str = (
            Player.player if self.player_move == Player.opponent else Player.opponent
        )

    def step(
        self, action: Tuple[int, str], verbose: bool = False
    ) -> Tuple[List[str], int, bool, Dict]:
        """Called by user - takes 2 turns, yours and your opponent's"""
        reward = self._step(action, verbose)
        if not self.done:
            opponent_action = self.opponent_choose_move(flatten_board(self.board))
            opponent_reward = self._step(opponent_action, verbose)
            # Negative sign is because the opponent's victory is your loss
            reward -= opponent_reward

        if verbose:
            if reward == 1:
                print("You win!")
            elif reward == -1:
                print("Oh no, your opponent won!")
            elif self.done:
                print("Game Drawn!")

        return flatten_board(self.board), reward, self.done, {}

    def _step(self, action: Tuple[int, str], verbose: bool = False) -> int:

        assert not self.done, "Game is done. Call reset() before taking further steps."

        position, counter = action
        row, col = convert_to_indices(position)

        assert (
            self.board[row][col] == Cell.EMPTY
        ), "You moved onto a square that already has a counter on it!"

        self.board = mark_square(self.board, row, col, counter)
        if verbose:
            print(f"{self.player_move} makes a move!")
            print(self)

        winner = is_winner(self.board)
        board_full = is_board_full(self.board)
        reward = 1 if winner else 0
        self.done = winner or board_full
        self.switch_player()

        return reward

    def reset(self, verbose: bool = False) -> Tuple[List[str], int, bool, Dict]:
        self.board = get_empty_board()

        self.done = False

        self.player_move = random.choice([Player.player, Player.opponent])
        self.went_first = self.player_move

        if verbose:
            print("Game starts!")
            print(self)

        if self.player_move == Player.opponent:
            opponent_action = self.opponent_choose_move(flatten_board(self.board))
            reward = -self._step(opponent_action, verbose)
        else:
            reward = 0

        return flatten_board(self.board), reward, self.done, {}


######## Do not worry about anything below here ###################


WIDTH = 600
HEIGHT = 600
LINE_WIDTH = 15
WIN_LINE_WIDTH = 15
BOARD_ROWS = 3
BOARD_COLS = 3
SQUARE_SIZE = 200
CIRCLE_RADIUS = 60
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = 55

RED = (255, 0, 0)
BG_COLOR = (20, 200, 160)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)


def flatten_board(board: List[List[str]]) -> List[str]:
    return [x for xs in board for x in xs]


def draw_non_board_elements(screen, game, player_move: str) -> None:
    draw_pieces(screen, game, player_move)


PLAYER_COLORS = {"player": "blue", "opponent": "red"}

counter_colors: Dict[Tuple[int, int], str] = {}

def draw_pieces(screen, game, player_move: str) -> None:
    # Draw circles and crosses based on board state
    
    # TODO: Fix this lol
    global counter_colors
    if flatten_board(game.board).count(" ") == 9:
        counter_colors = {}

    team_color = PLAYER_COLORS[player_move]

    board = game.board

    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            color = counter_colors.get((row, col), team_color)
            if board[row][col] == Cell.O:
                counter_colors[(row, col)] = color
                pygame.draw.circle(
                    screen,
                    color,
                    (
                        int(col * SQUARE_SIZE + SQUARE_SIZE // 2),
                        int(row * SQUARE_SIZE + SQUARE_SIZE // 2),
                    ),
                    CIRCLE_RADIUS,
                    CIRCLE_WIDTH,
                )
            elif board[row][col] == Cell.X:
                counter_colors[(row, col)] = color
                pygame.draw.line(
                    screen,
                    color,
                    (
                        col * SQUARE_SIZE + SPACE,
                        row * SQUARE_SIZE + SQUARE_SIZE - SPACE,
                    ),
                    (
                        col * SQUARE_SIZE + SQUARE_SIZE - SPACE,
                        row * SQUARE_SIZE + SPACE,
                    ),
                    CROSS_WIDTH,
                )
                pygame.draw.line(
                    screen,
                    color,
                    (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SPACE),
                    (
                        col * SQUARE_SIZE + SQUARE_SIZE - SPACE,
                        row * SQUARE_SIZE + SQUARE_SIZE - SPACE,
                    ),
                    CROSS_WIDTH,
                )


def check_and_draw_win(board: List, counter: str, screen: pygame.Surface, player_move: str) -> bool:

    for col in range(BOARD_COLS):
        if board[0][col] == counter and board[1][col] == counter and board[2][col] == counter:
            draw_vertical_winning_line(screen, col, counter, player_move)
            return True

    for row in range(BOARD_ROWS):
        if board[row][0] == counter and board[row][1] == counter and board[row][2] == counter:
            draw_horizontal_winning_line(screen, row, counter, player_move)
            return True

    if board[2][0] == counter and board[1][1] == counter and board[0][2] == counter:
        draw_asc_diagonal(screen, counter, player_move)
        return True

    if board[0][0] == counter and board[1][1] == counter and board[2][2] == counter:
        draw_desc_diagonal(screen, counter, player_move)
        return True

    return False


def draw_vertical_winning_line(screen, col, counter: str, player_move):
    posX = col * SQUARE_SIZE + SQUARE_SIZE // 2
    team_color = PLAYER_COLORS[player_move]

    pygame.draw.line(
        screen,
        team_color,
        (posX, 15),
        (posX, HEIGHT - 15),
        LINE_WIDTH,
    )


def draw_horizontal_winning_line(screen, row, counter, player_move):
    posY = row * SQUARE_SIZE + SQUARE_SIZE // 2

    team_color = PLAYER_COLORS[player_move]
    pygame.draw.line(
        screen,
        team_color,
        (15, posY),
        (WIDTH - 15, posY),
        WIN_LINE_WIDTH,
    )


def draw_asc_diagonal(screen, counter: str, player_move):
    team_color = PLAYER_COLORS[player_move]
    pygame.draw.line(
        screen,
        team_color,
        (15, HEIGHT - 15),
        (WIDTH - 15, 15),
        WIN_LINE_WIDTH,
    )


def draw_desc_diagonal(screen, counter: str, player_move):
    team_color = PLAYER_COLORS[player_move]
    pygame.draw.line(
        screen,
        team_color,
        (15, 15),
        (WIDTH - 15, HEIGHT - 15),
        WIN_LINE_WIDTH,
    )


def convert_to_indices(number: int) -> Tuple[int, int]:
    assert number in range(9), f"Output ({number}) not a valid number from 0 -> 8"
    return number // 3, number % 3


def render(
    choose_move: Callable[[List], Tuple[int, str]],
):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("TIC TAC TOE")
    screen.fill(BG_COLOR)

    # DRAW LINES
    pygame.draw.line(screen, LINE_COLOR, (0, SQUARE_SIZE), (WIDTH, SQUARE_SIZE), LINE_WIDTH)
    pygame.draw.line(
        screen,
        LINE_COLOR,
        (0, 2 * SQUARE_SIZE),
        (WIDTH, 2 * SQUARE_SIZE),
        LINE_WIDTH,
    )
    pygame.draw.line(screen, LINE_COLOR, (SQUARE_SIZE, 0), (SQUARE_SIZE, HEIGHT), LINE_WIDTH)
    pygame.draw.line(
        screen,
        LINE_COLOR,
        (2 * SQUARE_SIZE, 0),
        (2 * SQUARE_SIZE, HEIGHT),
        LINE_WIDTH,
    )

    game = WildTictactoeEnv()

    game_quit = False
    game_over = False
    player_move = random.choice([Player.player, Player.opponent])

    while not game_quit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.MOUSEBUTTONDOWN and game_over):
                game_quit = True

            if event.type == pygame.MOUSEBUTTONDOWN and not game_quit:

                if player_move == Player.player:
                    pos, counter = choose_move(flatten_board(game.board))
                else:
                    pos, counter = choose_move_randomly(flatten_board(game.board))

                row, col = convert_to_indices(pos)
                assert game.board[row][col] == Cell.EMPTY

                game.board = mark_square(game.board, row, col, counter)

                if check_and_draw_win(
                    game.board, Cell.X, screen=screen, player_move=player_move
                ) or check_and_draw_win(game.board, Cell.O, screen=screen, player_move=player_move):
                    game_over = True
                    print(f"{player_move} won!")
                draw_non_board_elements(screen, game, player_move)
                player_move = Player.player if player_move == Player.opponent else Player.opponent

        pygame.display.update()
