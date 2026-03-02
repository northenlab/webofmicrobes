from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def run(self, query: str, previous_result=None):
        pass
