# robot_arm_env.py
#
# Среда RobotArmEnv:
# - PyBullet-симуляция робота-манипулятора (KUKA iiwa из pybullet_data)
# - На столе лежит маленький кубик в случайном месте
# - Наблюдения: стек из нескольких grayscale-кадров с камеры (pixels only)
# - Действия: непрерывный вектор (Δx, Δy, Δz) для конца эффектора
# - Награда: shaped по расстоянию + усиленный бонус при приближении к цели
#   + SUCCESS по расстоянию и контакту

import os
import time
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import cv2


class RobotArmEnv(gym.Env):
    """
    Визуальная среда для обучения манипулятора по пикселям.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        img_size: int = 64,
        frame_stack: int = 4,
        frame_skip: int = 4,
        max_steps: int = 150,
    ):
        super().__init__()

        assert render_mode in (None, "human", "rgb_array")
        self.render_mode = render_mode

        # --- Параметры визуальной части ---
        self.img_size = img_size          # размер кадра (HxW)
        self.frame_stack = frame_stack    # сколько последних кадров стекуем
        self.frame_skip = frame_skip      # сколько шагов физики на одно действие
        self.max_steps = max_steps        # ограничение длины эпизода

        # --- Подключаемся к PyBullet ---
        if self.render_mode == "human":
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client_id)

        # --- Настройки камеры ---
        # Камера "сверху сбоку" (eye-to-hand)
        self.cam_target = [0.6, 0.0, 0.0]     # куда смотрим
        self.cam_distance = 1.2               # дистанция до цели
        self.cam_yaw = 90                     # поворот вокруг оси Z
        self.cam_pitch = -60                  # угол наклона
        self.cam_width = img_size
        self.cam_height = img_size

        # --- Пространство действий: Δx, Δy, Δz ---
        max_delta = 0.03
        self.action_space = spaces.Box(
            low=np.array([-max_delta, -max_delta, -max_delta], dtype=np.float32),
            high=np.array([max_delta, max_delta, max_delta], dtype=np.float32),
            dtype=np.float32,
        )

        # --- Пространство наблюдений: стек grayscale-кадров ---
        # shape: (frame_stack, H, W)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(frame_stack, img_size, img_size),
            dtype=np.uint8,
        )

        # --- Вспомогательные поля ---
        self.frames_buffer: list[np.ndarray] = []
        self.robot_id: Optional[int] = None
        self.table_id: Optional[int] = None
        self.obj_id: Optional[int] = None

        # индекс линка конца эффектора (для KUKA iiwa 7 DOF обычно 6)
        self.end_effector_link: int = 6

        self.step_counter: int = 0
        self.prev_dist: float = 0.0  # можно использовать для доп. shaping

        # начальная инициализация сцены
        self._setup_scene()

    # ======================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ДЛЯ СЦЕНЫ
    # ======================================================================

    def _setup_scene(self) -> None:
        """Полностью пересобрать сцену: плоскость, стол, робот, объект."""
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)

        # Плоскость
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        # Стол
        self.table_id = p.loadURDF(
            "table/table.urdf",
            basePosition=[0.7, 0.0, -0.65],
            useFixedBase=True,
            physicsClientId=self.client_id,
        )

        # Робот KUKA iiwa
        kuka_urdf = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")
        self.robot_id = p.loadURDF(
            kuka_urdf,
            basePosition=[0.0, 0.0, 0.0],
            useFixedBase=True,
            physicsClientId=self.client_id,
        )

        # Задаём стартовую позу (немного поднят над столом)
        self._reset_robot_joint_positions()

        # Объект-кубик в случайном месте на столе
        self._spawn_random_object()

        # Несколько шагов симуляции, чтобы всё “устаканилось”
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client_id)

    def _reset_robot_joint_positions(self) -> None:
        """Ставим робота в фиксированную начальную позу."""
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        # Простейшая поза: все нули, можно потом поменять под себя
        for j in range(num_joints):
            p.resetJointState(
                bodyUniqueId=self.robot_id,
                jointIndex=j,
                targetValue=0.0,
                targetVelocity=0.0,
                physicsClientId=self.client_id,
            )

    def _spawn_random_object(self) -> None:
        """Создаём маленький кубик в случайном месте на столе."""
        if self.obj_id is not None:
            try:
                p.removeBody(self.obj_id, physicsClientId=self.client_id)
            except Exception:
                # иногда PyBullet ругается "Remove body failed" — можно игнорить
                pass

        # Примерный центр стола около (0.7, 0.0, 0.0)
        x = 0.6 + np.random.uniform(-0.06, 0.06)
        y = np.random.uniform(-0.06, 0.06)
        z = 0.0  # при загрузке URDF высота скорректируется автоматически

        self.obj_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=[x, y, z],
            physicsClientId=self.client_id,
        )

    # ======================================================================
    # Доступ к позициям робота и объекта (для награды)
    # ======================================================================

    def _get_end_effector_pos(self) -> np.ndarray:
        """Позиция конца эффектора в мировых координатах."""
        state = p.getLinkState(
            self.robot_id,
            self.end_effector_link,
            physicsClientId=self.client_id,
        )
        pos = state[0]  # (x, y, z)
        return np.array(pos, dtype=np.float32)

    def _get_object_pos(self) -> np.ndarray:
        """Позиция объекта (центра массы) в мировых координатах."""
        pos, _ = p.getBasePositionAndOrientation(
            self.obj_id,
            physicsClientId=self.client_id,
        )
        return np.array(pos, dtype=np.float32)

    # ======================================================================
    # РАБОТА С КАМЕРОЙ И СТЕКАМИ КАДРОВ
    # ======================================================================

    def _get_camera_image(self) -> np.ndarray:
        """Сделать снимок с камеры и вернуть grayscale-кадр (H, W), uint8."""
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.cam_target,
            distance=self.cam_distance,
            yaw=self.cam_yaw,
            pitch=self.cam_pitch,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.cam_width) / self.cam_height,
            nearVal=0.1,
            farVal=3.0,
        )

        _, _, px, _, _ = p.getCameraImage(
            width=self.cam_width,
            height=self.cam_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.client_id,
        )

        # px: (H, W, 4) RGBA → берём только RGB, затем переводим в grayscale
        rgb = np.array(px, dtype=np.uint8)[:, :, :3]        # (H, W, 3)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)        # (H, W)
        gray = cv2.resize(
            gray,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_AREA,
        )
        return gray

    def _update_frames_buffer(self, new_frame: np.ndarray) -> np.ndarray:
        """
        Обновить буфер кадров (frame stack).
        new_frame: (H, W), uint8
        Возвращает стек: (frame_stack, H, W)
        """
        if len(self.frames_buffer) == 0:
            # При первом вызове дублируем один кадр N раз
            self.frames_buffer = [new_frame for _ in range(self.frame_stack)]
        else:
            self.frames_buffer.pop(0)
            self.frames_buffer.append(new_frame)

        stacked = np.stack(self.frames_buffer, axis=0)
        return stacked

    # ======================================================================
    # ОБЯЗАТЕЛЬНЫЕ МЕТОДЫ GYMNASIUM: reset / step / render / close
    # ======================================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Сброс среды в начальное состояние."""
        super().reset(seed=seed)

        if seed is not None:
            # синхронизируем случайности с seed
            np.random.seed(seed)

        self.step_counter = 0
        self.frames_buffer.clear()

        self._setup_scene()

        # Посчитаем начальное расстояние до объекта (если решим использовать diff)
        ee_pos = self._get_end_effector_pos()
        obj_pos = self._get_object_pos()
        self.prev_dist = float(np.linalg.norm(ee_pos - obj_pos))

        first_frame = self._get_camera_image()
        obs = self._update_frames_buffer(first_frame)

        info: Dict[str, Any] = {}
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Один шаг среды:
        - применяем Δx, Δy, Δz к позиции конца эффектора через IK
        - делаем несколько шагов физики (frame_skip)
        - считаем награду и признак завершения
        """
        self.step_counter += 1

        # гарантируем, что действие в допустимых пределах
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Текущая позиция конца эффектора
        ee_pos = self._get_end_effector_pos()
        target_pos = ee_pos + action  # новая желаемая позиция

        # Обратная кинематика: из целевой позиции → углы джоинтов
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_link,
            target_pos.tolist(),
            physicsClientId=self.client_id,
        )

        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)

        # Применяем joint_poses как целевые позиции
        for j in range(num_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(joint_poses[j]),
                force=200,
                physicsClientId=self.client_id,
            )

        # Frame skipping: несколько микрошагов физики на одно действие
        for _ in range(self.frame_skip):
            p.stepSimulation(physicsClientId=self.client_id)
            if self.render_mode == "human":
                time.sleep(1.0 / self.metadata["render_fps"])

        # Новое изображение и обновлённый стек кадров
        frame = self._get_camera_image()
        obs = self._update_frames_buffer(frame)

        # --------- РАСЧЁТ НАГРАДЫ ---------
        ee_pos = self._get_end_effector_pos()
        obj_pos = self._get_object_pos()
        dist = float(np.linalg.norm(ee_pos - obj_pos))

        # Проверяем, есть ли контакт между роботом и объектом
        contact_points = p.getContactPoints(
            bodyA=self.robot_id,
            bodyB=self.obj_id,
            physicsClientId=self.client_id,
        )
        contact = len(contact_points) > 0

        # ===== ВАРИАНТ 2 — усиленный shaping, чтобы дотягивался до цели =====
        reward = 0.0

        # 1) Нормализованная близость (0..1), растёт при уменьшении dist
        close_reward = 1.0 - np.tanh(2.5 * dist)
        reward += 4.0 * close_reward      # основной shaping по расстоянию

        # 2) Сильный дополнительный бонус, если мы уже близко (в зоне < 15 см)
        if dist < 0.15:
            reward += 4.0 * (0.15 - dist)

        # 3) Если робот близко и движется вниз — поощряем
        if dist < 0.12 and action[2] < 0:
            reward += 0.1

        # 4) SUCCESS по расстоянию — считаем, что дотянулись
        terminated = False
        if dist < 0.05:
            reward += 10.0
            terminated = True

        # 5) SUCCESS по контакту
        if contact:
            reward += 15.0
            terminated = True

        # 6) Штраф за шаг, чтобы эпизод не тянули бесконечно
        reward -= 0.01

        # --------- УСЛОВИЯ ЗАВЕРШЕНИЯ ЭПИЗОДА ---------
        truncated = False   # принудительное обрезание по длине/ограничениям

        # Эпизод слишком длинный → обрезаем
        if self.step_counter >= self.max_steps:
            truncated = True

        info: Dict[str, Any] = {
            "dist": dist,
            "contact": contact,
            "steps": self.step_counter,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """
        Рендер:
        - если render_mode="rgb_array", возвращаем кадр (H, W) grayscale.
        - если "human", всё уже показывается через GUI PyBullet.
        """
        if self.render_mode == "rgb_array":
            frame = self._get_camera_image()
            return frame
        return None

    def close(self) -> None:
        """Закрыть соединение с PyBullet."""
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)


# ============================
# (ОПЦИОНАЛЬНО) Глушим warning-и PyBullet
# ============================
# Если хочешь убрать "b3Warning / Remove body failed", можешь
# раскомментировать следующий блок и поставить его В САМОМ ВЕРХУ файла:
#
# import sys, os as _os
# sys.stderr = open(_os.devnull, "w")
#
# Но тогда ты не увидишь и traceback-ошибки в консоли.
