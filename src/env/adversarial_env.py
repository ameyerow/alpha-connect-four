import abc
from numpy import ndarray
from typing import Any, Tuple

class State:
    def __init__(self, board, current_player):
        """
        param board: The board of the game.
        param current_player: The current player of the game.
        """
        self.board = board
        self.current_player = current_player

class AdversarialEnv(metaclass=abc.ABCMeta):
    def __init__(self, state: State, action_space_shape, observation_space_shape):
        """
        param state: The current state of the environment, containing a board and a current
            player.
        param action_space_shape: The shape of the action space.
        param observation_space_shape: The shape of the observation space.
        """
        self.state: State = state
        self.action_space_shape = action_space_shape
        self.observation_space_shape = observation_space_shape

    @abc.abstractmethod
    def observation(self, state: State=None) -> ndarray:
        """
        Given the state of the environment, return the observation of the board from the 
        perspective of the current player. This essentially means that the board should
        be viewed as if the current player were "player 1". This should be in the form of
        a numpy array, as it will be passed into a model to predict action probabilities.

        return: A numpy array representing the observed environment.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, model, adversarial_model, state: State=None, render=False, test=False) -> float:
        """
        Given a model and an adversarial model, runs the environment to a terminal state using only NN estimation
        for move selection - used during MCTS rollout.

        param model: The model that dictates the moves of player 1
        param adversarial_model: The model that dictates the moves of player 2
        param state: The state of the environment. If None then this will use the state stored in the
            environment.
        return: The current player at the terminal state and the reward at the terminal state.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run_full_mcts(self, model, adversarial_model, state: State=None) -> Any:
        """
        Given a model and an adversarial model, run the environment to terminal state using a full MCTS evaluation for
        each moe selected - used during model evalulation.

        param model: The model that dictates the moves of player 1
        param adversarial_model: The model that dictates the moves of player 2
        param state: The state of the environment. If None then this will use the state stored in the
            environment.
        :return: The winning player at the terminal state
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action, state: State=None) -> Tuple[float, bool]:
        """
        Perform the given action on the state of the environment. 
        
        param action: The action that will be performed.
        param state: The state of the environment. If None then this will use the state stored in the
            environment.
        return: Who won the game (or 0 if its a draw) and whether the state is terminal.
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
        return: True if the action is legal, False otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_terminal_state(self, state: State=None) -> bool:
        """
        In the given environment state, determine if it's terminal. 

        param state: The state of the environment. If None then this will use the state stored in the
            environment.
        return: True if the state is terminal, False otherwise.
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

    @abc.abstractmethod
    def evaluate_state(self, state: State):
        """
        Evaluates a state and returns a list of rewards, with the first reward for player 1, the second for player 2 etc
        :param state: The state to be evaluated.
        :return: Returns a list of rewards, with the first reward for player 1, the second for player 2 etc
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_available_actions(self, state: State):
        """
        Evaluates a state and returns a list of available actions
        :param state: The state to be evaluated.
        :return: Returns a list of available actions
        """
        raise NotImplementedError
