import abc
import numpy as np
from typing import Any, Tuple
from collections import namedtuple

class State:
    def __init__(self, board, current_player):
        self.board = board
        self.current_player = current_player

class AdversarialEnv(metaclass=abc.ABCMeta):
    def __init__(self, state):
        self.state: State = state

    @abc.abstractmethod
    def observation(self, state: State=None) -> np.array:
        """
        Given the state of the environment, return the observation of the board from the 
        perspective of the current player. This essentially means that the board should
        be viewed as if the current player were "player 1". This should be in the form of
        a numpy array, as it will be passed into a model to predict action probabilities.

        return: A numpy array representing the observed environment.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, model, adversarial_model, state: State=None, render=False) -> Tuple[Any, float]:
        """
        Given a model and an adversarial model, runs the environment to a terminal state.

        param model: The model that dictates the moves of player 1
        param adversarial_model: The model that dictates the moves of player 2
        param state: The state of the environment. If None then this will use the state stored in the
            environment.
        return: The current player at the terminal state and the reward at the terminal state.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action, state: State=None) -> Tuple[float, bool]:
        """
        Perform the given action on the state of the environment. 
        
        param action: The action that will be performed.
        param state: The state of the environment. If None then this will use the state stored in the
            environment.
        return: The reward at the new state and whether it is terminal.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def perform_action_on_state(self, state: State, action) -> State:
        """
        Static method that performs the given action on the given state. Returns a new state without
        modifying the one passed into the function.

        param state: The state of the environment.
        param action: The action that will be performed.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_legal_action(self, action, state: State=None) -> bool:
        """
        In the given environment state, determine if an action is legal. 

        param action: The action that will be performed.
        param state: The state of the environment. If None then this will use the state stored in the
            environment.
        return: True if the acton is legal, False otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """
        Resets the environment.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def render(self, state: State = None):
        """
        Renders the environment.

        param state: The state of the environment. If None then this will use the state stored in the
            environment.
        """
        raise NotImplementedError

