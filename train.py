import numpy as np
import torch
from monte_carlo_tree_search import MCTS
from connect_4_gym import ConnectFourEnv
import random

class Train:

    def __init__(self, env: ConnectFourEnv, cur_model, next_model, num_episodes, test_games=50):
        self.env = env
        self.cur_model = cur_model
        self.next_model = next_model
        self.test_games = test_games
        self.num_episodes = num_episodes
        self.mcts = None
        random.seed(0)

    def execute_episode(self):
        self.env.reset()
        examples = []
        while True:
            current_player = self.env.current_player
            board_state = self.env.board

            self.mcts = MCTS(self.env, self.cur_model)
            pi = self.mcts.pi(board_state, current_player)
            examples.append([board_state, current_player, pi, None])
            action = np.random.choice(len(pi), p=pi)

            # reward for the player who placed the last piece
            next_board, reward = self.env.step(action)

            if reward is None:
                self.env.current_player *= -1
            else:
                for example in examples:

                    # multiplying the reward by the final player and the player at each step results in
                    # positive reward for the player who eventually wins at each step, and negative reward
                    # for the player who eventually loses at each step
                    example[3] = examples[1] * current_player * reward

            # we can either add the player as a feature in our NN or invert the board so that the NN always plays
            # from the position of player 1 (or 2)
            return examples

    def learn(self):
        all_examples = []
        for i in range(self.num_episodes):
            examples = self.execute_episode()
            all_examples.extend(examples)
        random.shuffle(all_examples)

        # copying the model
        self.save_checkpoint('cur_model', self.cur_model)
        self.load_checkpoint('cur_model', self.next_model)

        cur_model_mcts = MCTS(self.env, self.cur_model)

        self.next_model.train(all_examples)
        next_model_mcts = MCTS(self.env, self.next_model)

        # + for net cur_model_mcts victories, - for net next_model_mcts victories
        score = 0
        for i in range(int(self.test_games/2)):
            player, reward = self.env.run_game(cur_model_mcts.predict_move, next_model_mcts.predict_move)
            score += player * reward
            player, reward = self.env.run_game(next_model_mcts.predict_move, cur_model_mcts.predict_move)
            score -= player * reward

        if score > 0:
            self.cur_model = self.next_model





    def save_checkpoint(self, filename, model):
        torch.save({
            'state_dict': model.state_dict()
        }, filename)

    def load_checkpoint(self, filename, model):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])







