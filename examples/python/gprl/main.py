import pathlib
from typing import Literal

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import pygom
import sympy as sym
from moviepy.video.fx.MultiplySpeed import MultiplySpeed
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from pygom import *


def save_video(
    frames: list[np.ndarray],
    path: str | pathlib.Path,
    playback_speed: float = 2.0,
    logger: Literal["bar"] | None = None,
    **kwargs,
):
    """Saves frames as a mp4/gif"""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    video = ImageSequenceClip(frames, **kwargs)
    video = MultiplySpeed(playback_speed).apply(video)
    if path.name.endswith(".mp4"):
        video.write_videofile(path, logger=logger)
    elif path.name.endswith(".gif"):
        video.write_gif(path, logger=logger)
    else:
        raise ValueError(f"Unsupported video format '{path.name.split('.')[-1]}'")


class GPAlgorithm:
    def __init__(
        self,
        method: MethodBase,
        operators: list[OperatorBase] = [OpAdd(), OpSub(), OpMul(), OpDiv(), OpSin()],
        init: InitBase = HalfHalfInit(),
        erc_init_lb: float = -3.0,
        erc_init_ub: float = 3.0,
        expression_max_depth: int = 2,
        max_expression_size: int = 50,
    ):
        self.method = method
        self.operators = operators
        self.init = init
        self.erc_init_lb = erc_init_lb
        self.erc_init_ub = erc_init_ub
        self.max_depth = expression_max_depth
        self.max_expression_size = max_expression_size


class PolicyEvaluator:
    def __init__(
        self,
        env_id: str,
        ea: GPAlgorithm,
        num_episodes: int = 5,
        episode_length: int = 300,
        expression_size_objective: bool = False,
        seed: int | None = None,
        fix_episode_seeds: bool = True,
    ):
        if env_id.startswith("mo"):
            self.env = mo_gym.make(env_id, render_mode="rgb_array")
        else:
            self.env = gym.make(env_id, render_mode="rgb_array")

        # do a random step to determine the reward shape
        self.env.reset()
        reward = self.env.step(self.env.action_space.sample())[1]
        self.num_objectives = reward.shape[0] if isinstance(reward, np.ndarray) else 1

        self.num_total_objectives = self.num_objectives

        assert num_episodes > 0
        self.num_episodes = num_episodes
        assert episode_length > 0
        self.episode_length = episode_length
        self.expression_size_objective = expression_size_objective
        if self.expression_size_objective:
            self.num_total_objectives += 1

        self.num_actions, self.mapping = self._action_mapping(self.env.action_space)

        self.fix_episode_seeds = fix_episode_seeds
        self.set_seed(seed)

        self.frames = []
        self.render = False

        self.ea = ea

        ctx = GPContext(
            num_inputs=self.env.observation_space.shape[0],
            expression_template=Template(
                [
                    TemplateNode.full_nary(branching_factor=2, depth=self.ea.max_depth)
                    for _ in range(self.num_actions)
                ],
                [],
            ),
            operators=self.ea.operators,
            max_expression_size=self.ea.max_expression_size,
        )

        self.instance = PyGPInstance(
            ctx,
            self,
            self.num_total_objectives,
            self.ea.init,
            minimize=False,
            erc_init_lb=self.ea.erc_init_lb,
            erc_init_ub=self.ea.erc_init_ub,
        )

    def set_seed(self, seed: int | None):
        if seed is None:
            seed = int(np.random.default_rng().integers(2**32 - 1))
        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed

        self.episode_seeds = self.rng.integers(
            2**32 - 1, size=self.num_episodes
        ).tolist()

    def _action_mapping(self, action_space: gym.Space):
        """
        Returns the number of outputs and the mapping from output to action for a given gym 'space'
        """
        if isinstance(action_space, gym.spaces.Box):

            def map_continuous(actions: np.ndarray):
                return np.clip(
                    actions.astype(action_space.dtype),
                    min=action_space.low,
                    max=action_space.high,
                )

            return action_space.shape[0], map_continuous
        elif isinstance(action_space, gym.spaces.Discrete):

            def map_discrete(actions: np.ndarray):
                return np.argmax(actions, axis=1)

            return action_space.n, map_discrete
        else:
            raise ValueError(f"Unsupported action space '{type(action_space)}'")

    def render_frames(self, solution: SolutionBase) -> tuple[list[np.ndarray], int]:
        self.frames = []
        self.render = True

        s = AoSSet()
        s.add(solution)
        self.instance.evaluate_solutions(s)

        frames, self.frames = self.frames, []
        self.render = False
        return frames, self.env.metadata["render_fps"]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if hasattr(self, "instance"):
            del self.instance  # ensures the C++ is freed

    def run(
        self,
        budget: Budget = Budget(max_evaluations=10_000),
        seed=None,
        tracking_options: None | TrackingOptions = None,
    ):
        prev_seed = self.seed
        self.set_seed(seed)
        if tracking_options is not None:
            res = Tracked.run(
                self.instance,
                self.ea.method,
                budget,
                tracking_options,
                seed=self.rng.integers(2**32 - 1),
            )
        else:
            res = self.ea.method.run(
                self.instance,
                budget,
                seed=self.rng.integers(2**32 - 1),
            )

        self.set_seed(prev_seed)
        return res

    def __call__(self, predict, exprs, size):
        """The actual fitness function."""

        if not self.fix_episode_seeds:
            self.episode_seeds = self.rng.integers(
                2**32 - 1, size=self.num_episodes
            ).tolist()
        obs, info = self.env.reset(seed=self.episode_seeds[0])

        episode_lengths = [0]
        episode_rewards = [np.zeros(self.num_objectives)]
        while (
            len(episode_lengths) < self.num_episodes
            and episode_lengths[-1] < self.episode_length
        ):
            if self.render:
                self.frames.append(self.env.render())
            # get output(s) given the current state
            actions = predict(np.array([obs.tolist()]))

            # abort if the output is invalid
            if actions is None or not np.isfinite(actions).all():
                return np.repeat(np.inf, self.num_total_objectives), 1.0

            action = self.mapping(actions)[0]

            obs, reward, terminated, truncated, info = self.env.step(action)
            episode_lengths[-1] += 1
            episode_rewards[-1] += reward

            if terminated or truncated:
                if len(episode_lengths) >= self.num_episodes:
                    break
                obs, info = self.env.reset(
                    seed=self.episode_seeds[len(episode_lengths)]
                )
                episode_rewards.append(np.zeros(self.num_objectives))
                episode_lengths.append(0)

        total_reward = np.zeros(self.num_total_objectives)

        episode_rewards = np.vstack(episode_rewards)  # one reward per row

        total_reward[: self.num_objectives] = np.mean(episode_rewards, axis=0)

        if self.expression_size_objective:
            total_reward[-1] = -size  # not simplified

            # # sympy simplified (slow)
            # total_reward[-1] = -sum(
            #     sum(1 for _ in sym.preorder_traversal(sym.simplify(e, ratio=1.0)))
            #     for e in exprs
            # )

        # print(exprs, total_reward)
        return total_reward, 0.0


if __name__ == "__main__":
    # env = "Acrobot-v1"
    env = "CartPole-v1"
    # env = "Pendulum-v1"
    # env = "MountainCar-v0"
    # env = "MountainCarContinuous-v0"

    # env = "LunarLander-v3"

    # env = "mo-mountaincarcontinuous-v0"
    # env = "mo-lunar-lander-continuous-v3"

    ea = GPAlgorithm(
        pygom.classic.SimpleGA(),
        # MixedGOMEA(
        #     population_options=PopulationOptions(
        #         forced_improvements=True,
        #         donor_search_proportion=0.1,
        #         continuous_mutation_probability=1.0,
        #     ),
        #     ims_options=IMSOptions(
        #         initial_population_size=128,
        #         max_num_populations=8,
        #         restart_stale_populations=True,
        #     ),
        #     rv_options=RvOptions(enabled=False),
        # )
    )

    budget = Budget(max_time_seconds=300)  # 00)

    with PolicyEvaluator(env, ea, seed=None) as evaluator:
        archive, _ = evaluator.run(
            budget,
            tracking_options=TrackingOptions(
                "logs/baseline.csv",
                log_info=[
                    ("method_name", "baseline"),
                    ("env_id", env),
                    ("seed", str(evaluator.seed)),
                ],
                # log if a new best has been found
                report_on_archive_change=True,
            ),
        )

        print("Archive")
        for i in range(archive.size()):
            frames, fps = evaluator.render_frames(archive[i])
            save_video(frames, f"videos/solution_{i}.mp4", fps=fps)

            print(
                [
                    sym.simplify(e, ratio=1.0)
                    for e in evaluator.instance.format_solution(archive[i]).split(",")
                ],
                archive[i].quality().objectives,
            )
