
class ModelBackend:
    def __init__(self, model_name: str, use_mock_data: bool):
        self.model_name = model_name
        self.use_mock_data = use_mock_data

    def generate(self, prompt: str):
        if self.use_mock_data:
            return None  # mock image placeholder
        raise NotImplementedError(f"Configure generator for {self.model_name}.")
