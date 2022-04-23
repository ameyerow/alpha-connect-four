from abc import ABC, abstractmethod

class Player(ABC):

    def __init__(self, env: 'ConnectFourEnv', name='Player'):
        self.env = env
        self.name = name

    @abstractmethod
    def get_move(self, board):
        raise NotImplementedError()