from ast import Pass
from copy import copy
import math
from typing import List
import numpy as np

from adversarial_env import AdversarialEnv, State

# TODO: implement pi, which should generate the sample action probabilities - it should take in a board and player
# TODO: implement predict_move, which should select a move given board and player (and possibly env to get legal moves)
# predict_move and pi will likely be very similar / identical in logic

class Node:
    def __init__(self, parent, prior, state):
        self.prior: float = prior
        self.parent: Node = parent
        self.state: State = state

        self.visitation_count: int = 0
        self.total_value: int = 0
        self.children: List[Node] = []

    def expansion(self, model, env: AdversarialEnv):
        """
        Populates the Node's children states based on the actions available to it in the 
        environment.

        param model: An object (theoretically some pytorch model) whose forward function takes in a 
            current board state and returns a probability distribution over the action space and predicted 
            value for the terminal state.
        param env: The environment of the game.
        """
        action_probs, _ = model.forward(self.state)
        # TODO: balance probabilities based on some actions being illegal
        # if not env.is_legal_action(self.state, action):

        for action, action_prob in enumerate(action_probs):
            next_state = env.perform_action_on_state(self.state, self.current_player, action)
            child_node = Node(self, action_prob, next_state)
            self.children.append(child_node)
            
    def rollout(self, model) -> float:
        """
        Plays game forward from current state until arriving at some terminal state.

        param model:

        return:
        """
        terminal_player, reward = self.state.run(model, model)
        if terminal_player != self.current_player:
            reward *= -1
        return reward

    def backup(self, value: float):
        """
        Propogates the value of some terminal state (the result of a rollout) up the tree 
        through the parent pointers.

        param value:

        """
        self.total_value += value
        self.visitation_count += 1
        # Flips the value when calling backup because a positive reward for the current player is
        # a negative reward for the opposing player
        self.parent.backup(-value)

def UCB1(node: Node, parent_visitation_count: int, C: float = 2.0) -> float:
    """
    Calculates UCB1 score for a node

    param node: The node for which to calculate the UCB score
    param parent_visitation_count: The number of times the parent node has been visited
    param C: Exploration versus exploitation constant
    return: UCB1 score, a float. If the node has never been visited, UCB1=inf
    """
    if node.visitation_count == 0:
        return float("inf")

    average_value =  node.total_value / node.visitation_count
    visitation_ratio = math.log(parent_visitation_count)/node.visitation_count

    return average_value + node.prior * C * math.sqrt(visitation_ratio)

class MCTS:
    def __init__(self, env, model):
        pass

    def pi(self):
        pass

    def predict_move(self):
        pass