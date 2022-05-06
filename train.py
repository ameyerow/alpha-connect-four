import numpy as np
import torch
from mcts import MCTS
from connect_4_gym import ConnectFourEnv

class Train:

    def __init__(self, env: ConnectFourEnv, model, args):
        self.env = env
        self.model = model
        self.args = args
        self.mcts = None

    def execute_episode(self):
        self.env.reset()
        examples = []
        while True:
            current_player = self.env.current_player
            board_state = self.env.board

            self.mcts = MCTS(self.env, self.model, self.args)


