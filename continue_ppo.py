# continue_ppo.py
#
# Скрипт для ДОПОЛНИТЕЛЬНОГО обучения уже существующей PPO-модели.
#
# Идея:
#   - Мы уже обучили модель и сохранили её в ./models/ppo_robot_arm.zip
#     (или под другим именем).
#   - Теперь хотим "догнать" обучение ещё на N шагов, НЕ теряя предыдущие веса.
#
# ВАЖНО:
#   - Загрузка делается через PPO.load(...), а не через torch.load.
#   - Для правильных графиков в TensorBoard используем
#     reset_num_timesteps=False в model.learn(...).

import os
import torch
from stable_baselines3 import PPO

from utils import make_vec_env, ensure_dir
from vision_cnn import SmallCNN


def main():
    # =====================
    # 1. ПАПКИ
    # =====================
    LOG_DIR = "./logs/tensorboard/"
    MONITOR_DIR = "./logs/monitor/"
    MODEL_DIR = "./models/"

    # Имя модели БЕЗ ".zip"
    # Если у тебя файл ./models/ppo_robot_arm.zip,
    # то здесь нужно написать "ppo_robot_arm".
    MODEL_NAME = "ppo_robot_arm_cont"

    ensure_dir(LOG_DIR)
    ensure_dir(MONITOR_DIR)
    ensure_dir(MODEL_DIR)

    # =====================
    # 2. ПАРАМЕТРЫ СРЕДЫ (ДОЛЖНЫ СОВПАДАТЬ С train_ppo.py!)
    # =====================
    N_ENVS = 8
    IMG_SIZE = 64
    FRAME_STACK = 4
    FRAME_SKIP = 4
    MAX_STEPS = 150

    env = make_vec_env(
        n_envs=N_ENVS,
        seed=42,
        log_dir=MONITOR_DIR,
        vec_type="subproc",
        render=False,
        img_size=IMG_SIZE,
        frame_stack=FRAME_STACK,
        frame_skip=FRAME_SKIP,
        max_steps=MAX_STEPS,
    )

    # =====================
    # 3. ПАРАМЕТРЫ ПОЛИТИКИ (ДОЛЖНЫ БЫТЬ ТЕ ЖЕ, ЧТО И ПРИ ОБУЧЕНИИ)
    # =====================
    policy_kwargs = dict(
        features_extractor_class=SmallCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 256],
    )

    # =====================
    # 4. УСТРОЙСТВО
    # =====================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Используемое устройство:", device)

    # =====================
    # 5. ЗАГРУЗКА МОДЕЛИ
    # =====================
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    print(f"Загружаем модель из: {model_path}.zip")

    # custom_objects позволяет переопределить некоторые гиперпараметры
    # при загрузке модели, не ломая сохранённые веса.
    model = PPO.load(
        model_path,
        env=env,
        device=device,
        custom_objects=dict(
            learning_rate=1e-4,   # актуальная скорость обучения
            clip_range=0.2,
            target_kl=0.03,
        ),
    )

    # Обычно policy_kwargs уже сохранены внутри модели. Если хочешь явно
    # убедиться, что всё совпадает, можно оставить эту строку:
    model.policy_kwargs = policy_kwargs

    # =====================
    # 6. ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ
    # =====================
    MORE_TIMESTEPS = 1_000_000
    print(f"Продолжаем обучение ещё на {MORE_TIMESTEPS} шагов...")

    model.learn(
        total_timesteps=MORE_TIMESTEPS,
        tb_log_name="ppo_robot_arm_continue",
        reset_num_timesteps=False,  # ВАЖНО: не обнуляем счётчик шагов
    )

    # =====================
    # 7. СОХРАНЕНИЕ ОБНОВЛЁННОЙ МОДЕЛИ
    # =====================
    # Перезаписываем ту же модель (или можно сохранить под новым именем).
    model.save(model_path)
    print("Продолжение обучения завершено.")
    print("Модель обновлена и сохранена:", model_path)

    env.close()


if __name__ == "__main__":
    main()
