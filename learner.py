import numpy as np
import torch
from tqdm import tqdm
from adversarial_env import AdversarialEnv, State
from connect_four_env import ConnectFourEnv
from connect_two_model import ConnectTwoModel
from monte_carlo_tree_search import MCTS
from connect_four_model import ConnectFourModel
import random
import torch.nn as nn
from random_model import RandomModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Learner:

    def __init__(self, env: AdversarialEnv, cur_model, next_model, num_episodes, test_games=100):
        self.env = env
        self.cur_model = cur_model
        self.next_model = next_model
        self.test_games = test_games
        self.num_episodes = num_episodes
        self.mcts = None
        self.train_boards = []
        self.train_pis = []
        self.train_values = []
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

        print("Generating Episodes...")
        for _ in tqdm(range(self.num_episodes)):
            self.env.reset()
            boards, pis, values = self.execute_episode()
            
            self.train_boards.extend(boards)
            self.train_pis.extend(pis)
            self.train_values.extend(values)

        print("length of training set:", len(self.train_boards))

        # copying the model
        save_checkpoint('cur_model', self.cur_model)
        load_checkpoint('cur_model', self.next_model)

        train(self.next_model, self.train_boards, self.train_pis, self.train_values, batch_size=64, epochs=100)

        score, _, _, _ = compare_models(self.env, self.next_model, self.cur_model, self.test_games)

        if score > 0:
            self.cur_model = self.next_model
            print("new model was better with a net game lead of", score, "across", self.test_games, "games")
            self.train_boards = []
            self.train_values = []
            self.train_pis = []
        elif score == 0:
            print("new model was even with old model across", self.test_games, "games")
        else:
            print("new model was worse with net game losses of", score, "across", self.test_games, "games")


def compare_models(env, model1, model2, num_iters, render=False):
    print("Comparing models...")
    score = 0
    wins = 0
    losses = 0
    ties = 0
    for _ in tqdm(range(num_iters)):
        env.reset()
        winning_player = env.run(model1, model2, render=render)
        score += winning_player
        if winning_player > 0:
            wins += 1
        elif winning_player == 0:
            ties += 1
        else:
            losses += 1
        env.reset()
        winning_player = env.run(model2, model1, render=render)
        score -= winning_player
        if winning_player < 0:
            wins += 1
        elif winning_player == 0:
            ties += 1
        else:
            losses += 1

    if score > 0:
        print("model 1 was better with a net game lead of", score, "across", num_iters*2, "games")
    else:
        print("model 1 was worse with a net game loss of", score, "across", num_iters*2, "games")
    return score, wins, losses, ties


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
        print("Epoch loss:", epoch_loss/len(boards))


def pi_loss(sample_pis, predicted_pis):
    sample_pis = torch.from_numpy(sample_pis).to(device)
    loss = -(sample_pis * torch.log(predicted_pis)).sum(dim=1)
    return loss.mean()


def value_loss(sample_values, predicted_values):
    sample_values = torch.from_numpy(sample_values).to(device)
    loss = torch.sum((predicted_values.squeeze() - sample_values) ** 2)
    return loss.mean()

if __name__=="__main__":
    model = ConnectFourModel().to(device)
    # load_checkpoint("models/cur_model", model)
    learner = Learner(ConnectFourEnv(), model, model, 1)
    random_model = RandomModel(7).to(device)
    scores = []
    wins = []
    losses = []
    ties = []
    for i in range(20):
        learner.learn()
        score, win, loss, tie = compare_models(learner.env, learner.cur_model, random_model, 100)
        scores.append(score)
        wins.append(win)
        losses.append(loss)
        ties.append(tie)
    print(scores, wins, losses, ties)
    save_checkpoint("best_model", learner.cur_model)
    # env = ConnectFourEnv()
    # model = ConnectFourModel()
    # load_checkpoint("cur_model", model)
    # print(compare_models(env, model, RandomModel(7), 100))





    


