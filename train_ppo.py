# train_ppo.py
#
# Скрипт запуска обучения агента PPO на визуальной среде RobotArmEnv
# с использованием кастомной CNN (SmallCNN) как модуля "зрения".
#
# Основные шаги:
#   1. Создаём папки для логов и моделей.
#   2. Собираем векторную среду (несколько копий RobotArmEnv).
#   3. Настраиваем policy_kwargs (SmallCNN + архитектура голов).
#   4. Создаём модель PPO с нужными гиперпараметрами.
#   5. Запускаем model.learn(...) на заданное число шагов.
#   6. Сохраняем обученную модель.
#
# TensorBoard логируется в ./logs/tensorboard/, можно смотреть графики.

import os
import torch

from stable_baselines3 import PPO

from utils import make_vec_env, ensure_dir  # твои вспомогательные функции
from vision_cnn import SmallCNN


def main():
    # ================================
    # 1. ПАПКИ ДЛЯ ЛОГОВ И МОДЕЛЕЙ
    # ================================
    LOG_DIR = "./logs/tensorboard/"   # для TensorBoard
    MONITOR_DIR = "./logs/monitor/"   # для Monitor (эпизоды, награда)
    MODEL_DIR = "./models/"           # для сохранения весов

    ensure_dir(LOG_DIR)
    ensure_dir(MONITOR_DIR)
    ensure_dir(MODEL_DIR)

    # ================================
    # 2. ПАРАМЕТРЫ СРЕДЫ
    # ================================
    N_ENVS = 8            # число параллельных сред для векторизации
    IMG_SIZE = 64         # размер картинки камеры
    FRAME_STACK = 4       # сколько кадров подряд стекуем
    FRAME_SKIP = 4        # сколько шагов физики на одно действие
    MAX_STEPS = 150       # максимум шагов в эпизоде

    # ================================
    # 3. СОЗДАЁМ ВЕКТОРНУЮ СРЕДУ
    # ================================
    # make_vec_env — твоя функция, которая:
    #   - создаёт RobotArmEnv с нужными параметрами,
    #   - заворачивает его в VecEnv (DummyVecEnv или SubprocVecEnv),
    #   - при необходимости добавляет Monitor и VecFrameStack.
    env = make_vec_env(
        n_envs=N_ENVS,
        seed=42,
        log_dir=MONITOR_DIR,
        vec_type="subproc",   # "subproc" — параллельные процессы; "dummy" — 1 процесс
        render=False,
        img_size=IMG_SIZE,
        frame_stack=FRAME_STACK,
        frame_skip=FRAME_SKIP,
        max_steps=MAX_STEPS,
    )

    # ================================
    # 4. НАСТРОЙКА ПОЛИТИКИ (SmallCNN)
    # ================================
    # policy_kwargs передаётся в PPO, чтобы:
    #   - использовать SmallCNN как features_extractor,
    #   - задать размер вектора признаков (features_dim),
    #   - указать архитектуру полносвязных голов (policy/value).
    policy_kwargs = dict(
        features_extractor_class=SmallCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 256],   # по 2 слоя 256 нейронов для policy и value
    )

    # ================================
    # 5. ВЫБОР УСТРОЙСТВА (CPU / GPU)
    # ================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Используемое устройство:", device)

    # ================================
    # 6. СОЗДАЁМ МОДЕЛЬ PPO
    # ================================
    # Основные гиперпараметры:
    #   - n_steps: сколько шагов собираем перед одним обновлением
    #   - batch_size: размер батча для оптимизации
    #   - learning_rate: скорость обучения
    #   - gamma: discount factor
    #   - gae_lambda: параметр GAE (обобщённая оценка преимущества)
    #   - clip_range: порог клиппинга для PPO
    #   - target_kl: таргетное значение KL; если выше — шаг уменьшится
    model = PPO(
        policy="CnnPolicy",     # CnnPolicy ожидает (C, H, W) и CNN features extractor
        env=env,
        n_steps=1024,           # rollout длиной 1024 шага на каждую среду
        batch_size=2048,        # общий rollout: n_steps * N_ENVS = 1024 * 8 = 8192
                                # batch_size=2048 → 4 минибатча
        learning_rate=1e-4,     # небольшое lr для стабильности на картинках
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,           # без доп. энтропийного штрафа
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,         # если KL > 0.03, PPO уменьшит шаги
        verbose=1,
        tensorboard_log=LOG_DIR,
        device=device,
        policy_kwargs=policy_kwargs,
    )

    # ================================
    # 7. ЗАПУСК ОБУЧЕНИЯ
    # ================================
    TOTAL_TIMESTEPS = 1_000_000  # общее число шагов (по всем средам)

    print(f"Начинаем обучение на {TOTAL_TIMESTEPS} шагах...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name="ppo_robot_arm",  # имя "run-а" в TensorBoard
    )

    # ================================
    # 8. СОХРАНЕНИЕ МОДЕЛИ
    # ================================
    # Сохраняем модель без расширения — SB3 добавит .zip автоматически.
    model_path = os.path.join(MODEL_DIR, "ppo_robot_arm")
    model.save(model_path)

    print("✅ Обучение завершено.")
    print("Модель сохранена в:", model_path)

    env.close()


if __name__ == "__main__":
    main()
