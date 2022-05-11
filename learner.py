import numpy as np
import torch
from adversarial_env import AdversarialEnv, State
from connect_four_env import ConnectFourEnv
from monte_carlo_tree_search import MCTS
from connect_four_model import ConnectFourModel
import random
import torch.nn as nn
from random_model import RandomModel


class Learner:

    def __init__(self, env: AdversarialEnv, cur_model, next_model, num_episodes, test_games=100):
        self.env = env
        self.cur_model = cur_model
        self.next_model = next_model
        self.test_games = test_games
        self.num_episodes = num_episodes
        self.mcts = None
        random.seed(0)

    def execute_episode(self):
        examples = []
        while True:
            current_player = self.env.state.current_player
            board = self.env.state.board.copy()

            self.mcts = MCTS(self.env, self.cur_model)
            self.mcts.run()
            pi = self.mcts.pi()
            examples.append([board, current_player, pi])
            for action in range(len(pi)):
                if not self.env.is_legal_action(action):
                    pi[action] = 0
            prob_sum = np.sum(pi)
            pi /= np.sum(pi)

            action = np.random.choice(len(pi), p=pi)

            # reward for the player who placed the last piece
            winning_player, done = self.env.step(action)

            if done:
                boards = []
                pi = []
                values = []
                for example in examples:
                    # multiplying the reward by the final player and the player at each step results in
                    # positive reward for the player who eventually wins at each step, and negative reward
                    # for the player who eventually loses at each step

                    # (board, pi, v) form where pi is always from the perspective of player 1
                    boards.append(example[0] * example[1]) # changes board so current player is always player 1
                    pi.append(example[2]) # pi is unchanged
                    values.append(example[1] * winning_player) # winning player is multiplied by current player so always -1 if current player lost and 1 if current player won

                return boards, pi, values

    def learn(self):
        all_boards = []
        all_pis = []
        all_values = []
        for i in range(self.num_episodes):
            self.env.reset()
            print("creating episode", i)
            boards, pis, values = self.execute_episode()
            print("episode", i, "created")
            all_boards.extend(boards)
            all_pis.extend(pis)
            all_values.extend(values)

        print("length of training set:", len(all_boards))

        # copying the model
        save_checkpoint('cur_model', self.cur_model)
        load_checkpoint('cur_model', self.next_model)

        cur_model_mcts = MCTS(self.env, self.cur_model)

        train(self.next_model, all_boards, all_pis, all_values, batch_size=64, epochs=5)
        next_model_mcts = MCTS(self.env, self.next_model)

        # + for net cur_model_mcts victories, - for net next_model_mcts victories
        class MCTSLearner:
            def __init__(self, model):
                self.model = model

            def forward(self, observation_board):
                env = ConnectFourEnv()
                env.state = State(observation_board, 1)
                mcts = MCTS(env, self.model)
                mcts.run()
                return mcts.pi(), mcts.value()

        score = self.compare_models(self.cur_model, self.next_model, self.test_games)

        if score > 0:
            self.cur_model = self.next_model
            print("new model was better with a net game lead of", score, "across", self.test_games, "games")
        elif score == 0:
            print("new model was even with old model across", self.test_games, "games")
        else:
            print("new model was worse with net game losses of", score, "across", self.test_games, "games")

    def compare_models(self, model1, model2, num_iters):
        score = 0
        for i in range(num_iters):
            self.env.reset()
            winning_player = self.env.run(model1, model2)
            score += winning_player
            self.env.reset()
            winning_player = self.env.run(model2, model1)
            score -= winning_player

        print(score)
        if score > 0:
            print("model 1 was better with a net game lead of", score, "across", num_iters*2, "games")
        else:
            print("model 1 was worse with a net game loss of", score, "across", num_iters*2, "games")
        return score


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
    sample_pis = torch.from_numpy(sample_pis)
    loss = -(sample_pis * torch.log(predicted_pis)).sum(dim=1)
    return loss.mean()


def value_loss(sample_values, predicted_values):
    sample_values = torch.from_numpy(sample_values)
    loss = torch.sum((sample_values - predicted_values.view(-1)) ** 2)
    return loss.mean()

if __name__=="__main__":
    learner = Learner(ConnectFourEnv(), ConnectFourModel(), ConnectFourModel(), 5)
    scores = []
    for i in range(10):
        learner.learn()
        score = learner.compare_models(learner.cur_model, RandomModel(), 100)
        scores.append(score)
    print(scores)
    save_checkpoint("best_model", learner.cur_model)
    


