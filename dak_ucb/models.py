class ImageGeneratorBackend:
    def __init__(self, model_name: str, model_type: str, model_id: str, prior_id: str | None, use_mock_data: bool, device: str):
        self.model_name = model_name
        self.model_type = model_type
        self.model_id = model_id
        self.prior_id = prior_id
        self.use_mock_data = use_mock_data
        self.device = device
        self._pipeline = None
        self._prior = None

    def generate(self, prompt: str):
        if self.use_mock_data:
            return None
        if self.model_type == "sdxl":
            return self._generate_sdxl(prompt)
        if self.model_type == "kandinsky":
            return self._generate_kandinsky(prompt)
        raise NotImplementedError(f"Unknown image generator type: {self.model_type}")

    def _generate_sdxl(self, prompt: str):
        if self._pipeline is None:
            from diffusers import AutoPipelineForText2Image

            self._pipeline = AutoPipelineForText2Image.from_pretrained(
                self.model_id, torch_dtype=None
            ).to(self.device)
        image = self._pipeline(prompt=prompt).images[0]
        return image

    def _generate_kandinsky(self, prompt: str):
        if self._pipeline is None or self._prior is None:
            from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline

            if not self.prior_id:
                raise ValueError("Kandinsky requires a prior_id in config.")
            self._prior = KandinskyV22PriorPipeline.from_pretrained(self.prior_id).to(self.device)
            self._pipeline = KandinskyV22Pipeline.from_pretrained(self.model_id).to(self.device)
        prior_out = self._prior(prompt=prompt)
        image = self._pipeline(
            prompt=prompt,
            image_embeds=prior_out.image_embeds,
            negative_image_embeds=prior_out.negative_image_embeds,
        ).images[0]
        return image


class CaptioningBackend:
    def __init__(self, model_name: str, model_type: str, model_id: str, use_mock_data: bool, device: str):
        self.model_name = model_name
        self.model_type = model_type
        self.model_id = model_id
        self.use_mock_data = use_mock_data
        self.device = device
        self._model = None
        self._processor = None

    def caption(self, image):
        if self.use_mock_data:
            return ""
        if self.model_type in {"blip2", "instructblip"}:
            return self._caption_blip_family(image)
        if self.model_type == "llava":
            return self._caption_llava(image)
        raise NotImplementedError(f"Unknown captioning type: {self.model_type}")

    def _caption_blip_family(self, image):
        if self._model is None:
            from transformers import AutoProcessor, AutoModelForVision2Seq

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForVision2Seq.from_pretrained(self.model_id).to(self.device)
            self._model.eval()
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        out = self._model.generate(**inputs, max_new_tokens=32)
        return self._processor.batch_decode(out, skip_special_tokens=True)[0]

    def _caption_llava(self, image):
        if self._model is None:
            from transformers import AutoProcessor, LlavaForConditionalGeneration

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = LlavaForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
            self._model.eval()
        prompt = "Describe the image."
        inputs = self._processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        out = self._model.generate(**inputs, max_new_tokens=64)
        return self._processor.batch_decode(out, skip_special_tokens=True)[0]
