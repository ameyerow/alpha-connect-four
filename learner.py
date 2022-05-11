import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from adversarial_env import AdversarialEnv, State
from connect_four_env import ConnectFourEnv
from connect_two_model import ConnectTwoModel
from monte_carlo_tree_search import MCTS
from connect_four_model import ConnectFourModel
from big_connect_four_model import BigConnectFourModel
from long_connect_four_model import LongConnectFourModel
import random
import torch.nn as nn
from random_model import RandomModel
import pickle


device = torch.device("cuda:0")


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
        self.cur_optimizer = torch.optim.Adam(cur_model.parameters(), 0.001)
        self.next_optimizer = torch.optim.Adam(next_model.parameters(), 0.001)
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
        # if len(self.train_boards) == 0:
        #     with open("new_train", "rb") as tf:
        #         print("Loading previous training data")
        #         self.train_boards, self.train_pis, self.train_values = pickle.load(tf)
        for _ in tqdm(range(self.num_episodes)):
            self.env.reset()
            boards, pis, values = self.execute_episode()
            
            self.train_boards.extend(boards)
            self.train_pis.extend(pis)
            self.train_values.extend(values)
        while len(self.train_boards) > 512:
            self.train_boards.pop(0)
            self.train_pis.pop(0)
            self.train_values.pop(0)
        with open("new_train", "wb") as tf:
            pickle.dump((self.train_boards, self.train_pis, self.train_values), tf)

        print("length of training set:", len(self.train_boards))

        # copying the model
        save_checkpoint('cur_model', self.cur_model, self.cur_optimizer)
        self.next_model = load_checkpoint('cur_model', self.next_model, self.next_optimizer)

        self.next_model.to(device)

        train(self.next_model, self.train_boards, self.train_pis, self.train_values, optimizer=self.next_optimizer, batch_size=64, epochs=10)

        score, _, _, _ = compare_models(self.env, self.next_model, self.cur_model, self.test_games)

        if score > 0:
            self.cur_model = self.next_model
            self.cur_optimizer = self.next_optimizer
            print("new model was better with a net game lead of", score, "across", self.test_games, "games")
            # self.train_boards = []
            # self.train_values = []
            # self.train_pis = []
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


def save_checkpoint(filename, model, optimizer):
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, filename)


def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model


def train(model: nn.Module, boards, pis, values, batch_size, epochs, optimizer):
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

def graph_results(scores, wins, losses, ties):
    X = np.arange(len(scores))
    scores = np.array(scores)
    wins = np.array(wins)
    losses = np.array(losses)
    ties = np.array(ties)

    figure, axis = plt.subplots(2, 2)

    axis[0, 0].plot(X, scores)
    axis[0, 0].set_title("Net Scores")

    axis[0, 1].plot(X, wins, color="green")
    axis[0, 1].set_title("Wins")

    axis[1, 0].plot(X, losses, color="red")
    axis[1, 0].set_title("Losses")

    axis[1, 1].plot(X, ties, color="black")
    axis[1, 1].set_title("Ties")

    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

    from datetime import datetime
    dt = str(datetime.now()).replace(" ", "-")[-6:-1]
    plt.savefig(f'results/{dt}.png')
    plt.show()


if __name__=="__main__":
    # scores, wins, losses, ties = ([-16, 62, 16, 20, 52, 55, 82, 48, 52, 76, 54, 48, 57, 91, 40, 45, 98, 56, 76, 86, 55, 39, 39, 38, 102, 68, 81, 60, 64, 54, 53, 56, 62, 68, 70, 84, 86, 66, 56, 90, 48, 44, 54, 81, 68, 82, 69, 67, 80, 77, 69, 83, 79, 79, 82, 80, 85, 76, 98, 54, 88, 76, 90, 82, 75, 94, 90, 99, 84, 92, 94, 102, 80, 68, 84, 88, 85, 75, 90, 85, 105, 89, 102, 77, 84, 93, 101], [92, 131, 108, 110, 126, 127, 141, 124, 126, 138, 127, 124, 128, 145, 119, 122, 149, 128, 138, 142, 127, 119, 119, 119, 151, 134, 140, 130, 132, 127, 126, 128, 131, 133, 135, 142, 143, 133, 128, 145, 124, 122, 127, 140, 134, 141, 134, 133, 140, 138, 134, 141, 139, 139, 141, 140, 142, 138, 149, 127, 144, 138, 145, 141, 137, 147, 145, 149, 142, 146, 147, 151, 140, 133, 142, 144, 142, 137, 145, 142, 152, 144, 151, 138, 142, 146, 150], [108, 69, 92, 90, 74, 72, 59, 76, 74, 62, 73, 76, 71, 54, 79, 77, 51, 72, 62, 56, 72, 80, 80, 81, 49, 66, 59, 70, 68, 73, 73, 72, 69, 65, 65, 58, 57, 67, 72, 55, 76, 78, 73, 59, 66, 59, 65, 66, 60, 61, 65, 58, 60, 60, 59, 60, 57, 62, 51, 73, 56, 62, 55, 59, 62, 53, 55, 50, 58, 54, 53, 49, 60, 65, 58, 56, 57, 62, 55, 57, 47, 55, 49, 61, 58, 53, 49], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1])
    # graph_results(scores, wins, losses, ties)

    model1 = LongConnectFourModel()
    model2 = LongConnectFourModel()
    model1.to(device)
    model2.to(device)
    optimizer = torch.optim.Adam(model1.parameters(), 0.001)
    # load_checkpoint("models/long_best", model1, optimizer)
    learner = Learner(ConnectFourEnv(), model1, model2, 10)
    scores = []
    wins = []
    losses = []
    ties = []
    # with open("scores", "rb") as sf:
    #     scores, wins, losses, ties = pickle.load(sf)
    while True:
        learner.learn()
        score, win, loss, tie = compare_models(learner.env, learner.cur_model, RandomModel(7), 100)
        scores.append(score)
        wins.append(win)
        losses.append(loss)
        ties.append(tie)
        with open("scores", "wb") as sf:
            pickle.dump((scores, wins, losses, ties), sf)
        print(scores, wins, losses, ties)

    graph_results(scores, wins, losses, ties)
    # env = ConnectFourEnv()
    # model = LongConnectFourModel()
    # model2 = ConnectFourModel()
    # model.to(device)
    # model2.to(device)
    # load_checkpoint("cur_model", model)
    # load_checkpoint("models/best_connect_four_model", model2)
    # print(compare_models(env, model, RandomModel(7), 100, render=False))





    


