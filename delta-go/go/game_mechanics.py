import pickle
import sys

from typing import Any

from pathlib import Path

HERE = Path(__file__).parent.resolve()

# Hack as this won't pip install on replit
sys.path.append(str(HERE / "PettingZoo"))

import random
import time
from copy import deepcopy
from typing import Callable, Dict

import numpy as np
from gym.spaces import Box, Discrete
from torch import nn

from pettingzoo import go_v5
from pettingzoo.utils import BaseWrapper

BOARD_SIZE = 9
ALL_POSSIBLE_MOVES = np.arange(BOARD_SIZE**2 + 1)

# The komi to use is much debated. 7.5 seems to
# generalise well for different board sizes
# lifein19x19.com/viewtopic.php?f=15&t=17750
# 7.5 is also the komi used in alpha-go vs Lee Sedol
# (non-integer means there are no draws)

KOMI = 7.5


def play_go(
    your_choose_move: Callable,
    opponent_choose_move: Callable,
    game_speed_multiplier=1,
    render=True,
    verbose=False,
) -> None:

    env = GoEnv(
        opponent_choose_move,
        verbose=verbose,
        render=render,
        game_speed_multiplier=game_speed_multiplier,
    )

    observation, reward, done, info = env.reset()
    done = False
    while not done:
        action = your_choose_move(observation, info["legal_moves"])
        observation, reward, done, info = env.step(action)
    if render:
        time.sleep(100)
    return reward


class DeltaEnv(BaseWrapper):
    def __init__(
        self,
        env,
        opponent_choose_move: Callable,
        verbose: bool = False,
        render: bool = False,
        game_speed_multiplier: int = 1,
    ):

        super().__init__(env)

        self.opponent_choose_move = opponent_choose_move
        self.render = render
        self.verbose = verbose
        self.game_speed_multiplier = game_speed_multiplier
        self.action_space = Discrete(BOARD_SIZE**2 + 1)
        self.observation_space = Box(low=-1, high=1, shape=(BOARD_SIZE, BOARD_SIZE))
        self.num_envs = 1

    @property
    def turn(self) -> str:
        return self.env.agent_selection

    @property
    def turn_pretty(self) -> str:
        return self.turn[:-2]

    @property
    def observation(self):
        obs = self.env.last()[0]["observation"]
        player = obs[:, :, 0].astype("int")
        opponent = obs[:, :, 1].astype("int")
        return player - opponent

    @property
    def legal_moves(self):
        mask = self.env.last()[0]["action_mask"]
        full_space = self.env.action_space(self.turn)
        return np.arange(full_space.n)[mask.astype(bool)]

    @property
    def info(self) -> Dict:
        return {"legal_moves": self.legal_moves}

    @property
    def done(self) -> bool:
        return self.env.last()[2]

    def render_game(self) -> None:

        self.env.render()
        time.sleep(1 / self.game_speed_multiplier)

    def reset(self):
        super().reset()

        # Which color do we play as
        self.player = random.choice(["black_0", "white_0"])
        if self.verbose:
            print(
                f"Resetting Game.\nYou are playing with the {self.player[:-2]} tiles.\nBlack plays first\n\n"
            )

        if self.turn != self.player:
            self._step(
                self.opponent_choose_move(
                    observation=self.observation, legal_moves=self.legal_moves
                ),
            )

        return self.observation, 0, self.done, self.info

    def move_to_string(self, move: int):
        N = self.observation.shape[0]
        if move == N**2:
            return "passes"
        return f"places counter at coordinate: {(move//N, move%N)}"

    def __str__(self):
        return str(self.observation) + "\n"

    def _step(self, move: int) -> float:

        assert not self.done, "Game is done! Please reset() the env before calling step() again"
        assert move in self.legal_moves, f"{move} is an illegal move"

        if self.verbose:
            print(f"{self.turn_pretty} {self.move_to_string(move)}")

        self.env.step(move)

        if self.render:
            self.render_game()

        return self.reward

    @property
    def reward(self):
        return self.env.last()[1]

    def step(self, move: int):

        # Flipped because the env takes the step, changes the player, then we return the reward
        reward = -self._step(move)

        if not self.done:
            opponent_reward = self._step(
                self.opponent_choose_move(
                    observation=self.observation, legal_moves=self.legal_moves
                ),
            )
            # Flipped as above
            reward = opponent_reward

        if self.done and self.verbose:
            white_idx = int(self.turn_pretty == "white")
            black_idx = int(self.turn_pretty == "black")
            black_score = self.env.env.env.env.go_game.score()  # lol
            player_score = black_score if self.player == "black_0" else black_score * -1
            white_count = np.sum(self.env.last()[0]["observation"].astype("int")[:, :, white_idx])
            black_count = np.sum(self.env.last()[0]["observation"].astype("int")[:, :, black_idx])
            print(
                f"\nGame over. Reward = {reward}.\n"
                f"Player was playing as {self.player[:-2]}.\n"
                f"White has {white_count} counters.\n"
                f"Black has {black_count} counters.\n"
                f"Your score is {player_score}.\n"
            )

        return self.observation, reward, self.done, self.info


def GoEnv(
    opponent_choose_move: Callable[[np.ndarray, np.ndarray], int],
    verbose: bool = False,
    render: bool = False,
    game_speed_multiplier: int = 1,
) -> DeltaEnv:
    return DeltaEnv(
        go_v5.env(board_size=BOARD_SIZE, komi=KOMI),
        opponent_choose_move,
        verbose,
        render,
        game_speed_multiplier=game_speed_multiplier,
    )


def choose_move_randomly(observation, legal_moves):
    return legal_moves[int(random.random() * len(legal_moves))]


def choose_move_pass(observation, legal_moves) -> int:
    """passes on every turn."""
    return BOARD_SIZE**2


def transition_function(env: DeltaEnv, action: int) -> DeltaEnv:
    env = deepcopy(env)
    env._step(action)
    return env


def reward_function(env: DeltaEnv):
    return env.reward


def load_pkl(team_name: str, network_folder: Path = HERE) -> nn.Module:
    net_path = network_folder / f"{team_name}_file.pkl"
    assert (
        net_path.exists()
    ), f"Network saved using TEAM_NAME='{team_name}' doesn't exist! ({net_path})"
    with open(net_path, "rb") as handle:
        file = pickle.load(handle)
    return file


def save_pkl(file: Any, team_name: str) -> None:
    assert "/" not in team_name, "Invalid TEAM_NAME. '/' are illegal in TEAM_NAME"
    net_path = HERE / f"{team_name}_file.pkl"
    n_retries = 5
    for attempt in range(n_retries):
        try:
            with open(net_path, "wb") as handle:
                pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
            load_pkl(team_name)
            return
        except Exception:
            if attempt == n_retries - 1:
                raise
