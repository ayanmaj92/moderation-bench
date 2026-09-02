from abc import ABC, abstractmethod


class BasePrompt(ABC):
    """Abstract base for all prompt strategies."""

    def __init__(self, classes, descriptions=None):
        """
        Args:
            classes (list): list of class labels
            descriptions (dict): optional mapping {class: description}
        """
        self.classes = classes
        self.descriptions = descriptions or {}

    @abstractmethod
    def build_chat_messages(self, **kwargs):
        """Return the final prompt string.
        kwargs may include text, images, or other info."""
        pass
