from gym.spaces import Discrete
import warnings
import numpy as np
from gym.spaces import Box
import gym

from .env import ParallelEnv
from .env import AECIterable, AECIterator
from .utils import BaseWrapper

import io
import sys


class capture_stdout:
    """
    usage:

    with capture_stdout() as var:
        print("hithere")

        # value of var will be "hithere"
        data = var.getvalue()
    """

    def __init__(self):
        self.old_stdout = None

    def __enter__(self):
        self.old_stdout = sys.stdout
        self.buff = io.StringIO()
        sys.stdout = self.buff
        return self.buff

    def __exit__(self, type, value, traceback):
        sys.stdout = self.old_stdout
        self.buff.close()


class AssertOutOfBoundsWrapper(BaseWrapper):
    """
    this wrapper crashes for out of bounds actions
    Should be used for Discrete spaces
    """

    def __init__(self, env):
        super().__init__(env)
        assert all(
            isinstance(self.action_space(agent), Discrete)
            for agent in getattr(self, "possible_agents", [])
        ), "should only use AssertOutOfBoundsWrapper for Discrete spaces"

    def step(self, action):
        assert (action is None and self.dones[self.agent_selection]) or self.action_space(
            self.agent_selection
        ).contains(action), "action is not in action space"
        super().step(action)

    def __str__(self):
        return str(self.env)


class BaseParallelWraper(ParallelEnv):
    def __init__(self, env):
        self.env = env

        self.metadata = env.metadata
        try:
            self.possible_agents = env.possible_agents
        except AttributeError:
            pass

        # Not every environment has the .state_space attribute implemented
        try:
            self.state_space = self.env.state_space
        except AttributeError:
            pass

    def reset(self, seed=None, return_info=False, options=None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        if not return_info:
            res = self.env.reset(seed=seed, options=options)
            self.agents = self.env.agents
            return res
        else:
            res, info = self.env.reset(seed=seed, return_info=return_info, options=options)
            self.agents = self.env.agents
            return res, info

    def step(self, actions):
        res = self.env.step(actions)
        self.agents = self.env.agents
        return res

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def state(self):
        return self.env.state()

    @property
    def observation_spaces(self):
        warnings.warn(
            "The `observation_spaces` dictionary is deprecated. Use the `observation_space` function instead."
        )
        try:
            return {agent: self.observation_space(agent) for agent in self.possible_agents}
        except AttributeError as e:
            raise AttributeError(
                "The base environment does not have an `observation_spaces` dict attribute. Use the environments `observation_space` method instead"
            ) from e

    @property
    def action_spaces(self):
        warnings.warn(
            "The `action_spaces` dictionary is deprecated. Use the `action_space` function instead."
        )
        try:
            return {agent: self.action_space(agent) for agent in self.possible_agents}
        except AttributeError as e:
            raise AttributeError(
                "The base environment does not have an action_spaces dict attribute. Use the environments `action_space` method instead"
            ) from e

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)


class CaptureStdoutWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.metadata["render_modes"].append("ansi")

    def render(self, mode="human"):
        if mode == "ansi":
            with capture_stdout() as stdout:

                super().render("human")

                val = stdout.getvalue()
            return val
        else:
            return super().render(mode)

    def __str__(self):
        return str(self.env)


class ClipOutOfBoundsWrapper(BaseWrapper):
    """
    this wrapper crops out of bounds actions for Box spaces
    """

    def __init__(self, env):
        super().__init__(env)
        assert all(
            isinstance(self.action_space(agent), Box)
            for agent in getattr(self, "possible_agents", [])
        ), "should only use ClipOutOfBoundsWrapper for Box spaces"

    def step(self, action):
        space = self.action_space(self.agent_selection)
        if not (action is None and self.dones[self.agent_selection]) and not space.contains(action):
            assert (
                space.shape == action.shape
            ), f"action should have shape {space.shape}, has shape {action.shape}"
            action = np.clip(action, space.low, space.high)

        super().step(action)

    def __str__(self):
        return str(self.env)


class OrderEnforcingWrapper(BaseWrapper):
    """
    check all call orders:

    * error on getting rewards, dones, infos, agent_selection before reset
    * error on calling step, observe before reset
    * error on iterating without stepping or resetting environment.
    * warn on calling close before render or reset
    * warn on calling step after environment is done
    """

    def __init__(self, env):
        self._has_reset = False
        self._has_rendered = False
        self._has_updated = False
        super().__init__(env)

    def __getattr__(self, value):
        """
        raises an error message when data is gotten from the env
        which should only be gotten after reset
        """
        if value == "unwrapped":
            return self.env.unwrapped
        elif value == "possible_agents":
            print("error_possible_agents_attribute_missing(possible_agents)")
        elif value == "observation_spaces":
            raise AttributeError(
                "The base environment does not have an possible_agents attribute. Use the environments `observation_space` method instead"
            )
        elif value == "action_spaces":
            raise AttributeError(
                "The base environment does not have an possible_agents attribute. Use the environments `action_space` method instead"
            )
        elif value == "agent_order":
            raise AttributeError(
                "agent_order has been removed from the API. Please consider using agent_iter instead."
            )
        elif value in {
            "rewards",
            "dones",
            "infos",
            "agent_selection",
            "num_agents",
            "agents",
        }:
            raise AttributeError(f"{value} cannot be accessed before reset")
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{value}'")

    def render(self, mode="human"):
        if not self._has_reset:
            print("error_render_before_reset()")
        assert mode in self.metadata["render_modes"]
        self._has_rendered = True
        return super().render(mode)

    def step(self, action):
        if not self._has_reset:
            print("error_step_before_reset()")
        elif not self.agents:
            self._has_updated = True
            print("warn_step_after_done()")
            return None
        else:
            self._has_updated = True
            super().step(action)

    def observe(self, agent):
        if not self._has_reset:
            print("error_observe_before_reset()")
        return super().observe(agent)

    def state(self):
        if not self._has_reset:
            print("error_state_before_reset()")
        return super().state()

    def agent_iter(self, max_iter=2**63):
        if not self._has_reset:
            print("error_agent_iter_before_reset()")
        return AECOrderEnforcingIterable(self, max_iter)

    def reset(self, seed=None, return_info=False, options=None):
        self._has_reset = True
        self._has_updated = True
        super().reset(seed=seed, options=options)

    def __str__(self):
        if hasattr(self, "metadata"):
            return (
                str(self.env)
                if self.__class__ is OrderEnforcingWrapper
                else f"{type(self).__name__}<{str(self.env)}>"
            )
        else:
            return repr(self)


class AECOrderEnforcingIterable(AECIterable):
    def __iter__(self):
        return AECOrderEnforcingIterator(self.env, self.max_iter)


class AECOrderEnforcingIterator(AECIterator):
    def __next__(self):
        agent = super().__next__()
        assert self.env._has_updated, "need to call step() or reset() in a loop over `agent_iter`"
        self.env._has_updated = False
        return agent


class TerminateIllegalWrapper(BaseWrapper):
    """
    this wrapper terminates the game with the current player losing
    in case of illegal values

    parameters:
        - illegal_reward: number that is the value of the player making an illegal move.
    """

    def __init__(self, env, illegal_reward):
        super().__init__(env)
        self._illegal_value = illegal_reward
        self._prev_obs = None

    def reset(self, seed=None, return_info=False, options=None):
        self._terminated = False
        self._prev_obs = None
        super().reset(seed=seed, options=options)

    def observe(self, agent):
        obs = super().observe(agent)
        if agent == self.agent_selection:
            self._prev_obs = obs
        return obs

    def step(self, action):
        current_agent = self.agent_selection
        if self._prev_obs is None:
            self.observe(self.agent_selection)
        assert (
            "action_mask" in self._prev_obs
        ), "action_mask must always be part of environment observation as an element in a dictionary observation to use the TerminateIllegalWrapper"
        _prev_action_mask = self._prev_obs["action_mask"]
        self._prev_obs = None
        if self._terminated and self.dones[self.agent_selection]:
            self._was_done_step(action)
        elif not self.dones[self.agent_selection] and not _prev_action_mask[action]:
            self._cumulative_rewards[self.agent_selection] = 0
            self.dones = {d: True for d in self.dones}
            self._prev_obs = None
            self.rewards = {d: 0 for d in self.dones}
            self.rewards[current_agent] = float(self._illegal_value)
            self._accumulate_rewards()
            self._dones_step_first()
            self._terminated = True
        else:
            super().step(action)

    def __str__(self):
        return str(self.env)
