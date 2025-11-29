# utils.py
#
# Вспомогательные функции для проекта:
# - make_env: фабрика одной среды RobotArmEnv
# - make_vec_env: создание векторной среды (DummyVecEnv или SubprocVecEnv)
# - ensure_dir: создание папок для логов/моделей

from __future__ import annotations

import os
from typing import Callable, Optional, List

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from robot_arm_env import RobotArmEnv


def ensure_dir(path: str) -> None:
    """Создать директорию, если её ещё нет."""
    if path is not None and path != "" and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def make_env(
    env_id: str = "RobotArmEnv",
    seed: int = 0,
    rank: int = 0,
    log_dir: Optional[str] = None,
    render: bool = False,
    img_size: int = 64,
    frame_stack: int = 4,
    frame_skip: int = 4,
    max_steps: int = 150,
) -> Callable[[], gym.Env]:
    """
    Фабрика одной среды.

    Возвращает функцию _init, которую можно передать в DummyVecEnv/SubprocVecEnv.

    Параметры:
    - seed: базовый сид (к нему добавляется rank, чтобы среды были разными)
    - rank: индекс процесса/среды
    - log_dir: куда писать монитор-логи (episode_rewards, episode_lengths)
    - render: если True, создаётся RobotArmEnv с render_mode="human"
    """

    def _init() -> gym.Env:
        # Выбираем режим рендера
        render_mode = "human" if render else None

        env = RobotArmEnv(
            render_mode=render_mode,
            img_size=img_size,
            frame_stack=frame_stack,
            frame_skip=frame_skip,
            max_steps=max_steps,
        )

        # Устанавливаем сид для среды
        env.reset(seed=seed + rank)

        # Оборачиваем в Monitor для логов
        if log_dir is not None:
            ensure_dir(log_dir)
            env = Monitor(env, filename=os.path.join(log_dir, f"env_{rank}"))
        else:
            env = Monitor(env)

        return env

    return _init


def make_vec_env(
    n_envs: int = 4,
    seed: int = 0,
    log_dir: Optional[str] = "./logs/monitor/",
    vec_type: str = "dummy",
    render: bool = False,
    img_size: int = 64,
    frame_stack: int = 4,
    frame_skip: int = 4,
    max_steps: int = 150,
) -> gym.Env:
    """
    Создаёт векторную среду (DummyVecEnv или SubprocVecEnv).

    Параметры:
    - n_envs: сколько параллельных сред
    - seed: базовый сид
    - log_dir: папка для логов Monitor
    - vec_type: "dummy" или "subproc"
    - render: для отладки можно сделать одну среду с GUI
    """

    # Для обучения обычно render=False, GUI сильно замедляет.
    # Для отладки можно сделать n_envs=1 и render=True.

    env_fns: List[Callable[[], gym.Env]] = []
    for i in range(n_envs):
        env_fns.append(
            make_env(
                seed=seed,
                rank=i,
                log_dir=log_dir,
                render=render if (render and i == 0) else False,
                img_size=img_size,
                frame_stack=frame_stack,
                frame_skip=frame_skip,
                max_steps=max_steps,
            )
        )

    if vec_type == "subproc":
        vec_env = SubprocVecEnv(env_fns)
    else:
        # по умолчанию используем DummyVecEnv — проще дебажить
        vec_env = DummyVecEnv(env_fns)

    return vec_env
