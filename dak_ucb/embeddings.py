import numpy as np
import torch


class EmbeddingBackend:
    def __init__(
        self,
        prompt_dim: int,
        image_dim: int,
        use_mock_data: bool,
        clip_model_id: str,
        dinov2_model_id: str,
        device: str,
        seed: int | None = None,
    ):
        self.prompt_dim = prompt_dim
        self.image_dim = image_dim
        self.use_mock_data = use_mock_data
        self.clip_model_id = clip_model_id
        self.dinov2_model_id = dinov2_model_id
        self.device = device
        self.rng = np.random.default_rng(seed)

        self._clip = None
        self._clip_processor = None
        self._dino = None
        self._dino_processor = None

    def encode_prompt(self, prompt: str) -> np.ndarray:
        if self.use_mock_data:
            return self._randn(self.prompt_dim)
        clip, processor = self._load_clip()
        inputs = processor(text=[prompt], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = clip.get_text_features(**inputs)
        return self._normalize(outputs.squeeze(0).cpu().numpy())

    def encode_image(self, image) -> np.ndarray:
        if self.use_mock_data:
            return self._randn(self.image_dim)
        dino, processor = self._load_dino()
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = dino(**inputs).last_hidden_state
        pooled = outputs.mean(dim=1)
        return self._normalize(pooled.squeeze(0).cpu().numpy())

    def encode_image_clip(self, image) -> np.ndarray:
        if self.use_mock_data:
            return self._randn(self.prompt_dim)
        clip, processor = self._load_clip()
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = clip.get_image_features(**inputs)
        return self._normalize(outputs.squeeze(0).cpu().numpy())

    def encode_target(self, target) -> np.ndarray:
        if self.use_mock_data:
            return self._randn(self.image_dim)
        return self.encode_image(target)

    def _load_clip(self):
        if self._clip is None:
            from transformers import CLIPModel, CLIPProcessor

            self._clip = CLIPModel.from_pretrained(self.clip_model_id).to(self.device)
            self._clip.eval()
            self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_id)
        return self._clip, self._clip_processor

    def _load_dino(self):
        if self._dino is None:
            from transformers import AutoImageProcessor, AutoModel

            self._dino = AutoModel.from_pretrained(self.dinov2_model_id).to(self.device)
            self._dino.eval()
            self._dino_processor = AutoImageProcessor.from_pretrained(self.dinov2_model_id)
        return self._dino, self._dino_processor

    def _randn(self, dim: int) -> np.ndarray:
        v = self.rng.standard_normal((dim,))
        return v / (np.linalg.norm(v) + 1e-12)

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)
