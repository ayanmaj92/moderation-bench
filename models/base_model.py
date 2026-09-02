from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, model_name: str, model_size=None, device="cuda:0"):
        self.model_name = model_name
        self.model_size = model_size
        self.model = None
        self.processor = None
        self.device = device

    @abstractmethod
    def load(self):
        """Load model + processor."""
        pass
