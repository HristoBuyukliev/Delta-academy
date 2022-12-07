import math
import random
import time
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import gym
import pygame

ARENA_WIDTH = 15
ARENA_HEIGHT = 15
BLOCK_SIZE = 50

assert ARENA_HEIGHT == ARENA_WIDTH, "current only support square arenas"

TAIL_STARTING_LENGTH = 1

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# will break with >6 teams
BIKE_COLORS = [
    (237, 0, 3),
    (53, 0, 255),
    (1, 254, 1),
    (255, 134, 0),
    (255, 254, 55),
    (140, 0, 252),
]

HERE = Path(__file__).parent.resolve()


@dataclass
class State:
    player: "Bike"
    opponent: "Bike"
    player_move: Optional[int] = None

    @property
    def bikes(self) -> Tuple["Bike", "Bike"]:
        return self.player, self.opponent

    @property
    def state_id(self) -> Tuple[Tuple, Tuple, Optional[int]]:
        return self.player.bike_state, self.opponent.bike_state, self.player_move

    @property
    def bike_to_move(self) -> "Bike":
        return self.player if self.player_move is None else self.opponent

    def copy(self) -> "State":
        return State(copy(self.player), copy(self.opponent), self.player_move)


def choose_move_randomly(state: State) -> int:
    """This works but the bots die very fast."""
    return int(random.random() * 3) + 1


def rules_rollout(state: State) -> int:
    """Rollout policy that tries not to hit anything."""
    obstacles = (
        (
            [(ARENA_HEIGHT - 1, i) for i in range(ARENA_WIDTH)]
            + [(i, ARENA_WIDTH - 1) for i in range(ARENA_HEIGHT)]
            + [(i, 0) for i in range(ARENA_HEIGHT)]
            + [(0, i) for i in range(ARENA_WIDTH)]
        )
        + state.player.positions
        + state.opponent.positions
    )

    poss_actions = get_possible_actions()
    while len(poss_actions) > 0:
        action = poss_actions[math.floor(random.random() * len(poss_actions))]
        bike_moving = copy(state.bike_to_move)
        bike_moving.take_action(action)
        if bike_moving.head not in obstacles:
            return action
        else:
            poss_actions.remove(action)
    return 1


def choose_move_square(state: State) -> int:
    """This bot happily goes round the edge in a square."""

    orientation = state.player.direction
    head = state.player.head

    if orientation == 0 and head[1] <= 3:
        return 3
    if orientation == 3 and head[0] <= 3:
        return 3
    if orientation == 2 and head[1] >= ARENA_HEIGHT - 3:
        return 3
    if orientation == 1 and head[0] >= ARENA_WIDTH - 3:
        return 3
    return 1


def play_tron(
    your_choose_move: Callable,
    opponent_choose_move: Callable,
    game_speed_multiplier: float = 1.0,
    render: bool = True,
    verbose: bool = False,
) -> float:
    env = TronEnv(
        opponent_choose_move=opponent_choose_move,
        verbose=verbose,
        render=render,
        game_speed_multiplier=game_speed_multiplier,
    )

    state, reward, done, _ = env.reset()
    done = False

    return_ = 0
    while not done:
        action = your_choose_move(state)
        state, reward, done, _ = env.step(action)
        return_ += reward
        if verbose:
            print("Done", done, "is_terminal", is_terminal(state), "state:", state.state_id)

    return return_


def in_arena(pos: Tuple[int, int]) -> bool:
    y_out = pos[1] <= 0 or pos[1] >= ARENA_HEIGHT - 1
    x_out = pos[0] <= 0 or pos[0] >= ARENA_WIDTH - 1
    return not x_out and not y_out


class Action:
    """The action taken by the bike.

    The bike has 3 options:
        1. Go forward
        2. Turn left (and go forward 1 step)
        3. Turn right (and go forward 1 step)
    """

    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


class Orientation:
    """Direction the bike is pointing."""

    SOUTH = 0  # negative y-direction
    EAST = 1  # positive x-direction
    NORTH = 2  # positive y-direction
    WEST = 3  # negative x-direction


class Bike:
    def __init__(
        self, name: str = "bike", starting_position: Optional[Tuple[int, int]] = None
    ) -> None:
        # Initial orientation of the bike is chosen at random
        # TODO: What if we start at the edge of the arena?
        #  Surely we shouldn't be facing the edge? Or away from the edge (tail would be outside arena)?
        self.direction = random.choice(
            [Orientation.EAST, Orientation.WEST, Orientation.NORTH, Orientation.SOUTH]
        )

        if starting_position is None:
            bike_head_x = random.randint(ARENA_WIDTH // 4, 3 * ARENA_WIDTH // 4)
            bike_head_y = random.randint(ARENA_HEIGHT // 4, 3 * ARENA_HEIGHT // 4)
        else:
            bike_head_x, bike_head_y = starting_position

        for offset in range(1, TAIL_STARTING_LENGTH + 1):
            bike_tail_x = (
                bike_head_x - offset
                if self.direction == Orientation.EAST
                else bike_head_x + offset
                if self.direction == Orientation.WEST
                else bike_head_x
            )
            bike_tail_y = (
                bike_head_y - offset
                if self.direction == Orientation.NORTH
                else bike_head_y + offset
                if self.direction == Orientation.SOUTH
                else bike_head_y
            )
            self.positions = [(bike_head_x, bike_head_y), (bike_tail_x, bike_tail_y)]

        self.alive = True
        self.name = name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bike):
            raise NotImplementedError
        return self.name == other.name

    def __copy__(self) -> "Bike":
        positions = list(self.positions)
        cls = self.__class__
        new_copy = cls.__new__(cls)
        new_copy.alive = self.alive
        new_copy.name = self.name
        new_copy.direction = self.direction
        new_copy.positions = positions
        return new_copy

    def __repr__(self) -> str:
        return f"Bike {self.name}"

    def set_positions(self, positions: List[Tuple[int, int]]) -> None:
        self.positions = positions

    def has_hit_boundaries(self) -> bool:
        return not in_arena(self.head)

    def has_hit_self(self) -> bool:
        return self.head in self.body

    def kill_bike(self) -> None:
        self.alive = False

    @property
    def length(self) -> int:
        return len(self.positions)

    @property
    def head(self) -> Tuple[int, int]:
        return self.positions[0]

    @property
    def bike_state(self) -> Tuple:
        """Describes fully the state of a bike.

        Can be used as a dictionary key
        """
        return tuple(self.positions) if self.alive else ("dead",)

    @property
    def body(self) -> List[Tuple[int, int]]:
        return self.positions[1:]

    def take_action(self, action: int) -> None:
        assert action in {1, 2, 3}

        if action == 2:
            new_orientation = (self.direction + 1) % 4
        elif action == 3:
            new_orientation = (self.direction - 1) % 4
        else:
            new_orientation = self.direction

        x, y = self.head
        if new_orientation % 2 == 0:
            # South is 0 (y -= 1), North is 2 (y += 1)
            y += new_orientation - 1
        else:
            # East is 1 (x += 1), West is 3 (x -= 1)
            x += 2 - new_orientation

        # Update position and orientation
        if action is not None:
            self.positions.insert(0, (x, y))
            self.direction = new_orientation

    def remove_tail_end(self) -> None:
        del self.positions[-1]


def get_starting_positions() -> List[Tuple[int, int]]:
    """Get a list of starting positions that are not too close together."""

    min_x = ARENA_WIDTH // 4
    max_x = 3 * ARENA_WIDTH // 4
    min_y = ARENA_HEIGHT // 4
    max_y = 3 * ARENA_HEIGHT // 4
    positions = []

    # Return n**2 points
    n = 3

    offset_x = (max_x - min_x) // (n - 1)
    offset_y = (max_y - min_y) // (n - 1)
    for i in range(n):
        for j in range(n):
            positions.append((min_x + offset_x * i, min_y + offset_y * j))

    return positions


class TronEnv(gym.Env):

    SCREEN_WIDTH = ARENA_WIDTH * BLOCK_SIZE
    SCREEN_HEIGHT = ARENA_HEIGHT * BLOCK_SIZE

    def __init__(
        self,
        opponent_choose_move: Callable,
        verbose: bool = False,
        render: bool = False,
        game_speed_multiplier: float = 1.0,
    ):

        self.opponent_choose_move = opponent_choose_move
        self._render = render
        self.verbose = verbose
        self.game_speed_multiplier = game_speed_multiplier

        self.starting_positions = get_starting_positions()
        self.score = 0
        if self._render:
            self.init_visuals()

    def reset(self) -> Tuple[State, int, bool, Dict]:
        self.num_steps_taken = 0

        random.shuffle(self.starting_positions)

        player = Bike(name="player", starting_position=self.starting_positions[0])
        opponent = Bike(name="opponent", starting_position=self.starting_positions[1])
        self.state = State(player, opponent)

        self.dead_bikes: List[Bike] = []
        assert len(self.bikes) == 2

        self.color_lookup = dict(zip([bike.name for bike in self.bikes], BIKE_COLORS))
        return self.state.copy(), 0, False, {}

    @property
    def bikes(self) -> Tuple[Bike, Bike]:
        return self.state.bikes

    @property
    def done(self) -> bool:
        return is_terminal(self.state)

    def _step(self, action: int) -> None:
        if action not in [Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT]:
            raise ValueError(f"Invalid action: {action}")

        if self.state.player_move is None:
            self.state.player_move = action
            return

        self.state.player.take_action(self.state.player_move)
        self.state.opponent.take_action(action)
        self.state.player_move = None

        for bike in self.bikes:
            if self.has_hit_tails(bike.head) or bike.has_hit_boundaries():
                bike.kill_bike()
        self.head_to_head_collision()

        if self.verbose:
            print(f"{self.num_steps_taken} steps taken")

    def head_to_head_collision(self) -> None:
        if self.state.player.head == self.state.opponent.head:
            self.state.player.kill_bike()
            self.state.opponent.kill_bike()

    def has_hit_tails(self, bike_head: Tuple[int, int]) -> bool:
        return any(bike_head in other_bike.body for other_bike in self.bikes)

    @staticmethod
    def boundary_elements_mask(matrix: np.ndarray) -> np.ndarray:
        mask = np.ones(matrix.shape, dtype=bool)
        mask[matrix.ndim * (slice(1, -1),)] = False
        return mask

    def get_opponent_state(self) -> State:
        return State(
            player=self.state.opponent,
            opponent=self.state.player,
        )

    def step(self, action: int) -> Tuple[State, int, bool, Dict]:
        # Step the player's bike if it's not dead (tournament)
        self._step(action)

        bike_state = self.get_opponent_state()
        action = self.opponent_choose_move(state=bike_state)
        self._step(action)

        if self._render:
            self.render_game()
            time.sleep(1 / self.game_speed_multiplier)

        self.num_steps_taken += 1

        reward = 0
        if self.done:
            winner = self.find_winner()
            if winner is not None:
                reward = 1 if winner == self.state.player else -1
        return self.state.copy(), reward, self.done, {}

    def find_winner(self) -> Optional[Bike]:
        assert self.done
        if all(not bike.alive for bike in self.bikes):
            return None
        return self.state.player if self.state.player.alive else self.state.opponent

    def init_visuals(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(
            (ARENA_WIDTH * BLOCK_SIZE, ARENA_HEIGHT * BLOCK_SIZE)  # , pygame.FULLSCREEN
        )
        pygame.display.set_caption("Tron")
        self.clock = pygame.time.Clock()
        self.screen.fill(WHITE)
        self.score_font = pygame.font.SysFont("comicsansms", 35)

    def render_game(self, screen: Optional[pygame.Surface] = None) -> None:

        # If no injected screen, use graphical constants
        if screen is None:
            screen = self.screen
            screen_width = self.SCREEN_WIDTH
            screen_height = self.SCREEN_HEIGHT
            block_size = BLOCK_SIZE
        else:  # Overwrite  visual consts based on screen
            screen_width = screen.get_width()
            screen_height = screen.get_height()
            block_size = screen_width // ARENA_WIDTH  # Assume square

        screen.fill(WHITE)

        # Draw boundaries
        pygame.draw.rect(
            screen, BLACK, [1, 1, screen_width - 1, screen_height - 1], width=block_size
        )

        for bike in self.bikes:

            color = self.color_lookup[bike.name]

            for bike_pos in bike.body:
                bike_y = (
                    ARENA_HEIGHT - bike_pos[1] - 1
                )  # Flip y axis because pygame counts 0,0 as top left
                pygame.draw.rect(
                    screen,
                    color,
                    [bike_pos[0] * block_size, bike_y * block_size, block_size, block_size],
                )
            # Flip y axis because pygame counts 0,0 as top left
            bike_y = ARENA_HEIGHT - bike.head[1] - 1
            pygame.draw.rect(
                screen,
                BLACK,
                [
                    bike.head[0] * block_size,
                    bike_y * block_size,
                    block_size,
                    block_size,
                ],
            )

        # This may cause flashing in the tournament
        pygame.display.update()


def human_player(*args: Any, **kwargs: Any) -> int:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                return 3
            if event.key == pygame.K_LEFT:
                return 2
    return 1


def transition_function(state: State, action: int, make_copies: bool = True) -> State:
    """The transition function return a new State after an action has been taken in the current
    state.

    Tron requires both players to move simultaneously. The first time this function is called it
    should be called with the action of the player. This move is stored but the player is not moved.
    The second time this function is called it should be called with the action of the opponent.
    This time both players are moved. If make_copies is set to True the state is copied before the
    action is taken to avoid mutating the original state.
    """

    if make_copies:
        state = State(copy(state.player), copy(state.opponent), state.player_move)

    if state.player_move is None:
        state.player_move = action
        return state

    state.opponent.take_action(action)
    state.player.take_action(state.player_move)
    state.player_move = None

    for bike in state.bikes:
        if has_hit_tails(bike.head, state) or bike.has_hit_boundaries():
            bike.kill_bike()

    head_to_head_collision(state)

    return state


def reward_function(successor_state: State) -> int:
    player_dead = not successor_state.player.alive
    opponent_dead = not successor_state.opponent.alive
    if player_dead and not opponent_dead:
        return -1
    elif not player_dead and opponent_dead:
        return 1
    return 0


def has_hit_tails(bike_head: Tuple[int, int], state: State) -> bool:
    return any(bike_head in bike.body for bike in state.bikes)


def head_to_head_collision(state: State) -> None:
    """Kill bikes involved in head-to-head collisions."""
    if state.opponent.head == state.player.head:
        state.opponent.kill_bike()
        state.player.kill_bike()


def is_terminal(successor_state: State) -> bool:
    return any(not bike.alive for bike in successor_state.bikes)


def get_possible_actions():
    return [1, 2, 3]
