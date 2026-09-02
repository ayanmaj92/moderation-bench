from abc import ABC, abstractmethod

class BaseDataset(ABC):
    def __init__(self, path, target_labels, label_to_category):
        self.path = path
        self.target_labels = target_labels
        self.label_to_category = label_to_category

    @abstractmethod
    def load_all(self):
        """Return texts, labels, images for the full dataset."""
        pass
