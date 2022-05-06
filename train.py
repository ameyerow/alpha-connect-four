import numpy as np
import torch
from monte_carlo_tree_search import MCTS
from connect_4_gym import ConnectFourEnv
import random

class Train:

    def __init__(self, env: ConnectFourEnv, cur_model, next_model, args):
        self.env = env
        self.cur_model = cur_model
        self.next_model = next_model
        self.args = args
        self.mcts = None
        random.seed(0)

    def execute_episode(self):
        self.env.reset()
        examples = []
        while True:
            current_player = self.env.current_player
            board_state = self.env.board

            self.mcts = MCTS(self.env, self.cur_model, self.args)
            pi = self.mcts.pi(board_state, current_player)
            examples.append([board_state, current_player, pi, None])
            action = np.random.choice(len(pi), p=pi)
            next_board, reward = self.env.step(action)

            if reward is None:
                self.env.current_player *= -1
            else:
                for example in examples:
                    example[3] = examples[1] * current_player * reward
            return examples

    def learn(self):
        all_examples = []
        for i in range(self.args['episodes']):
            examples = self.execute_episode()
            all_examples.extend(examples)
        random.shuffle(all_examples)

        # copying the model
        self.save_checkpoint('cur_model', self.cur_model)
        self.load_checkpoint('cur_model', self.next_model)

        cur_model_mcts = MCTS(self.env, self.cur_model, self.args)

        self.next_model.train(all_examples)
        next_model_mcts = MCTS(self.env, self.next_model, self.args)



    def save_checkpoint(self, filename, model):
        torch.save({
            'state_dict': model.state_dict()
        }, filename)

    def load_checkpoint(self, filename, model):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])







