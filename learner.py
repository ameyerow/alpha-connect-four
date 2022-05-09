import numpy as np
import torch
from adversarial_env import AdversarialEnv, State
from connect_four_env import ConnectFourEnv
from monte_carlo_tree_search import MCTS
import random
import torch.nn as nn


class Learner:

    def __init__(self, env: AdversarialEnv, cur_model, next_model, num_episodes, test_games=50):
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
            current_player = self.env.state.current_player
            board_state = self.env.state.board

            self.mcts = MCTS(self.env, self.cur_model)
            pi = self.mcts.pi(board_state, current_player)
            examples.append([board_state, current_player, pi])
            action = np.random.choice(len(pi), p=pi)

            # reward for the player who placed the last piece
            reward, done = self.env.step(action)

            if not done:
                self.env.current_player *= -1
            else:
                boards = []
                pi = []
                values = []
                for example in examples:
                    # multiplying the reward by the final player and the player at each step results in
                    # positive reward for the player who eventually wins at each step, and negative reward
                    # for the player who eventually loses at each step

                    # (board, pi, v) form where pi is always from the perspective of player 1
                    boards.append(example[0] * example[1])
                    pi.append(example[2])
                    values.append(example[1] * current_player * reward)

                return boards, pi, values

    def learn(self):
        all_boards = []
        all_pis = []
        all_values = []
        for i in range(self.num_episodes):
            boards, pis, values = self.execute_episode()
            all_boards.extend(boards)
            all_pis.extend(pis)
            all_values.extend(values)

        # copying the model
        self.save_checkpoint('cur_model', self.cur_model)
        self.load_checkpoint('cur_model', self.next_model)

        cur_model_mcts = MCTS(self.env, self.cur_model)

        train(self.next_model, all_boards, all_pis, all_values, batch_size=64, epochs=10)
        next_model_mcts = MCTS(self.env, self.next_model)

        # + for net cur_model_mcts victories, - for net next_model_mcts victories
        class MCTSLearner:
            def __init__(self, model):
                self.model = model

            def forward(self, X):
                current_player
                env = ConnectFourEnv()
                env.state = State(X, current_player)
                mcts = MCTS(env, self.model)
                mcts.run()
                return mcts.pi(), mcts.value()

        score = 0
        for i in range(int(self.test_games / 2)):
            player, reward = self.env.run(cur_model_mcts.predict_move, next_model_mcts.predict_move)
            score += player * reward
            player, reward = self.env.run(next_model_mcts.predict_move, cur_model_mcts.predict_move)
            score -= player * reward

        if score > 0:
            self.cur_model = self.next_model


def save_checkpoint(filename, model):
    torch.save({
        'state_dict': model.state_dict()
    }, filename)


def load_checkpoint(filename, model):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])


def train(model: nn.Module, boards, pis, values, batch_size, epochs):
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    boards = np.array(boards)
    pis = np.array(pis)
    values = np.array(values)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        p = np.random.permutation(len(boards))
        boards = boards[p]
        pis = pis[p]
        values = values[p]
        i = 0
        while i * batch_size < len(boards):
            boards_batch = boards[i * batch_size:(i + 1) * batch_size]
            pis_batch = pis[i * batch_size:(i + 1) * batch_size]
            values_batch = values[i * batch_size:(i + 1) * batch_size]

            optimizer.zero_grad()
            predicted_pis, predicted_values = model.forward(boards_batch)
            loss = pi_loss(pis_batch, predicted_pis) + value_loss(values_batch, predicted_values)
            loss.backward()
            epoch_loss += loss.item() * len(boards_batch)
            optimizer.step()
            i += 1
        print("Epoch loss:", epoch_loss)


def pi_loss(sample_pis, predicted_pis):
    loss = -(sample_pis * torch.log(predicted_pis)).sum(dim=1)
    return loss.mean()


def value_loss(sample_values, predicted_values):
    loss = torch.sum((sample_values - predicted_values.view(-1)) ** 2)
    return loss.mean()
