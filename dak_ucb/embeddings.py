import numpy as np


class EmbeddingBackend:
    def __init__(self, prompt_dim: int, image_dim: int, use_mock_data: bool, seed: int | None = None):
        self.prompt_dim = prompt_dim
        self.image_dim = image_dim
        self.use_mock_data = use_mock_data
        self.rng = np.random.default_rng(seed)

    def encode_prompt(self, prompt: str) -> np.ndarray:
        if self.use_mock_data:
            return self._randn(self.prompt_dim)
        raise NotImplementedError("Configure prompt encoder (e.g., CLIP text encoder).")

    def encode_image(self, image) -> np.ndarray:
        if self.use_mock_data:
            return self._randn(self.image_dim)
        raise NotImplementedError("Configure image encoder (e.g., DINOv2).")

    def encode_image_clip(self, image) -> np.ndarray:
        if self.use_mock_data:
            return self._randn(self.prompt_dim)
        raise NotImplementedError("Configure CLIP image encoder.")

    def encode_target(self, target) -> np.ndarray:
        if self.use_mock_data:
            return self._randn(self.image_dim)
        raise NotImplementedError("Configure reference/target encoder.")

    def _randn(self, dim: int) -> np.ndarray:
        v = self.rng.standard_normal((dim,))
        return v / (np.linalg.norm(v) + 1e-12)
