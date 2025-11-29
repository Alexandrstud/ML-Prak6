# vision_cnn.py
#
# Здесь описана небольшая свёрточная сеть SmallCNN, которая используется
# как модуль "зрения" (features_extractor) в Stable-Baselines3.
#
# Идея:
#   - На вход подаётся наблюдение из среды: стек кадров (C, H, W),
#     где C = frame_stack (например, 4), H=W=64 или 84.
#   - Через 3 сверточных слоя мы извлекаем пространственные признаки.
#   - Потом делаем Flatten → Linear → ReLU и получаем вектор фиксированного
#     размера features_dim (по умолчанию 256).
#
# В SB3 этот features_extractor подключается через policy_kwargs:
#   policy_kwargs = dict(
#       features_extractor_class=SmallCNN,
#       features_extractor_kwargs=dict(features_dim=256),
#   )

from typing import Tuple

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SmallCNN(BaseFeaturesExtractor):
    """
    Небольшая CNN для обработки стека кадров с камеры.

    Вход:
        observation_space: Box с формой (C, H, W), dtype=uint8,
                           где C = число каналов (frame_stack).

    Выход:
        Вектор признаков размерности features_dim (по умолчанию 256),
        который дальше идёт в политику и value-функцию (головы).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
    ) -> None:
        """
        Параметры:
            observation_space: описание наблюдения из среды (Box (C, H, W)).
            features_dim: размер выходного вектора признаков.

        Важно:
        - Сначала вызываем super().__init__(observation_space, features_dim),
          чтобы BaseFeaturesExtractor знал размерность выхода.
        """
        super().__init__(observation_space, features_dim)

        assert isinstance(observation_space, gym.spaces.Box), \
            "SmallCNN работает только с Box наблюдениями"

        # Сохраняем форму наблюдения: (C, H, W)
        self._obs_shape: Tuple[int, int, int] = observation_space.shape
        n_input_channels = self._obs_shape[0]  # C = frame_stack

        # Свёрточная часть сети — похожа на классическую Nature DQN архитектуру.
        # Здесь мы:
        #   - уменьшаем пространственное разрешение,
        #   - увеличиваем число каналов (кол-во "карт признаков").
        self.cnn = nn.Sequential(
            # Первый слой:
            # - большие фильтры 8x8,
            # - stride=4, чтобы быстро уменьшить H и W.
            nn.Conv2d(
                in_channels=n_input_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
            ),
            nn.ReLU(),

            # Второй слой:
            # - фильтры 4x4, stride=2 для дальнейшего уменьшения размера.
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
            ),
            nn.ReLU(),

            # Третий слой:
            # - маленькие фильтры 3x3, stride=1, для более локальных признаков.
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),

            # Разворачиваем всё в один длинный вектор (batch_size, n_flatten)
            nn.Flatten(),
        )

        # Чтобы корректно задать линейный слой, нужно знать,
        # сколько элементов получается после conv-части (n_flatten).
        with torch.no_grad():
            # Создаём "фиктивный" тензор размера (1, C, H, W)
            sample_input = torch.zeros(1, *self._obs_shape)
            # Пропускаем через cnn и смотрим размер второго измерения
            n_flatten = self.cnn(sample_input).shape[1]

        # Полносвязная часть:
        # - Линейное преобразование n_flatten → features_dim
        # - Нелинейность ReLU
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход сети.

        Параметры:
            observations: тензор формы (batch_size, C, H, W),
                          который приходит от SB3-политики.

        Возвращает:
            тензор формы (batch_size, features_dim) — вектор признаков для
            каждого элемента батча.
        """
        x = self.cnn(observations)
        x = self.linear(x)
        return x
