from abc import ABC, abstractmethod


class Decider(ABC):

    #
    # Decides which action to take for any given state
    # Returns an Action object
    #
    @abstractmethod
    def decide(self, state):
        pass
