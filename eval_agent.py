# eval_agent.py
#
# Запуск обученного агента PPO в GUI PyBullet (режим демонстрации).
# - Открывается окно PyBullet
# - Агент действует детерминированно (без случайности)
# - Можно наблюдать, как рука тянется к объекту

import os
import time

from stable_baselines3 import PPO

from robot_arm_env import RobotArmEnv


def main():
    MODEL_DIR = "./models/"
    model_path = os.path.join(MODEL_DIR, "ppo_robot_arm_cont")

    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"Не найден файл модели: {model_path}.zip\n"
            f"Сначала запусти train_ppo.py, чтобы обучить и сохранить модель."
        )

    print("Загружаем модель из:", model_path + ".zip")
    model = PPO.load(model_path)

    # Создаём ОДНУ среду с GUI
    env = RobotArmEnv(
        render_mode="human",
        img_size=64,
        frame_stack=4,
        frame_skip=4,
        max_steps=150,
    )

    NUM_EPISODES = 5
    print(f"Запускаем {NUM_EPISODES} демонстрационных эпизодов...")

    try:
        for ep in range(NUM_EPISODES):
            obs, info = env.reset()
            done = False
            truncated = False
            ep_reward = 0.0

            print(f"\n=== Эпизод {ep + 1}/{NUM_EPISODES} ===")

            while not (done or truncated):
                # deterministic=True → без ε-greedy, "чистая" политика
                action, _ = model.predict(obs, deterministic=True)

                obs, reward, done, truncated, info = env.step(action)
                ep_reward += reward

                # немного притормозим, чтобы движения были видны
                #time.sleep(1.0 / 60.0)

            print(
                f"Эпизод {ep + 1} завершён. "
                f"total_reward = {ep_reward:.3f}, "
                f"steps = {info.get('steps', 'N/A')}, "
                f"contact = {info.get('contact', False)}"
            )

    except KeyboardInterrupt:
        print("\nОстановка по Ctrl+C.")

    finally:
        env.close()
        print("Окно PyBullet закрыто.")


if __name__ == "__main__":
    main()
