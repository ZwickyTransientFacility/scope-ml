"""
    Defines:
        - Abstract AbstractClassifier class. Individual classifiers should be subclasses thereof.

"""

from abc import ABC, abstractmethod
from collections import defaultdict


class AbstractClassifier(ABC):
    def __init__(self, name):
        # classifier name: label_classifier
        self.name = name
        # model that will be trained and evaluated
        self.model = None
        # metadata needed to set up the classifier
        self.meta = defaultdict(str)

    @abstractmethod
    def setup(self, **kwargs):
        pass

    @abstractmethod
    def load(self, **kwargs):
        pass

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass

    @abstractmethod
    def save(self, **kwargs):
        pass
