from typing import Optional


class ImageGeneratorBackend:
    def __init__(self, model_name: str, model_type: str, model_id: str, prior_id: Optional[str], use_mock_data: bool, device: str):
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
            import torch
            from diffusers import StableDiffusionXLPipeline

            self._pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model_id, torch_dtype=torch.float16
            ).to(self.device)
        image = self._pipeline(prompt=prompt).images[0]
        return image

    def _generate_kandinsky(self, prompt: str):
        if self._pipeline is None:
            import torch
            from diffusers import AutoPipelineForText2Image

            self._pipeline = AutoPipelineForText2Image.from_pretrained(
                self.model_id, torch_dtype=torch.float16
            ).to(self.device)
            self._pipeline.to(torch.bfloat16)
        image = self._pipeline(prompt=prompt).images[0]
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
        import torch

        if self.model_type == "instructblip":
            if self._model is None:
                from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

                self._processor = InstructBlipProcessor.from_pretrained(self.model_id)
                self._model = InstructBlipForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                ).to(self.device)
                self._model.eval()
            prompt = "Generate a concise caption describing only what is visually apparent in this image in one short sentence."
            inputs = self._processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            ).to(self.device, torch.float16)
            out = self._model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.1,
            )
            return self._processor.decode(out[0], skip_special_tokens=True).strip()

        if self._model is None:
            from transformers import Blip2ForConditionalGeneration, Blip2Processor

            self._processor = Blip2Processor.from_pretrained(self.model_id)
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto" if self.device == "cuda" else None,
            )
            self._model.eval()
        prompt = "Question: Briefly describe this image. Answer:"
        inputs = self._processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device, torch.float16)
        out = self._model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=0.7,
        )
        full_output = self._processor.decode(out[0], skip_special_tokens=True)
        return full_output.replace(prompt, "").strip()

    def _caption_llava(self, image):
        if self._model is None:
            import torch
            from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

            self._processor = LlavaNextProcessor.from_pretrained(self.model_id)
            self._model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self._model.eval()
        prompt = "Generate a concise caption describing only what is visually apparent in this image in one short sentence."
        prompt_template = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = self._processor(
            text=prompt_template,
            images=image,
            return_tensors="pt",
        ).to(self.device, torch.float16)
        out = self._model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            temperature=0.1,
        )
        full_output = self._processor.batch_decode(out, skip_special_tokens=True)[0]
        return full_output.split("ASSISTANT:")[-1].strip()
