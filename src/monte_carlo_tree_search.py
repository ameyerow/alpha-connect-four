import math
import torch
import numpy as np
from typing import List
from numpy import ndarray
from copy import deepcopy

from .env.adversarial_env import AdversarialEnv, State


class Node:
    def __init__(self, action: int, parent, prior: float, state: State):
        """
        param action: The action taken on the parent Node to arrive at this state.
        param parent: The parent node.
        param prior: The action probability associated with arriving at this node from the parent node.
        param state: The environment state associated with this node.
        """
        self.action: int = action
        self.parent: Node = parent
        self.prior: float = prior
        self.state: State = state
        self.responsible_player = state.current_player * -1

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
        # Check if state is already terminal, don't expand if so
        if env.is_terminal_state(self.state):
            return

        action_probs, _ = model.forward(env.observation(self.state))
        action_probs = action_probs.squeeze().detach().cpu().numpy()

        # Balance probabilities based on some actions being illegal
        for action in range(len(action_probs)):
            if not env.is_legal_action(action, state=self.state):
                action_probs[action] = 0
        prob_sum = np.sum(action_probs)
        if prob_sum == 0:
            action_probs = np.ones(env.action_space_shape) / len(action_probs) # TODO: maybe change divisor
        else:
            action_probs /= np.sum(action_probs)

        # Create a child node for each legal state
        for action, action_prob in enumerate(action_probs):
            if not env.is_legal_action(action, state=self.state):
                continue
            next_state = env.perform_action_on_state(self.state, action)
            child_node = Node(action, self, action_prob, next_state)
            self.children.append(child_node)
            
    def rollout(self, model, env: AdversarialEnv) -> float:
        """
        Plays game forward from current state until arriving at some terminal state.

        param model: An object (theoretically some pytorch model) whose forward function takes in a 
            current board state and returns a probability distribution over the action space and predicted 
            value for the terminal state.
        param env: The environment of the game.
        return: The value at the terminal state from the perspective of the current node.
        """
        winning_player = env.run(model, model, state=deepcopy(self.state))
        if winning_player == 0:
            return 0
        elif winning_player != self.responsible_player:
            return -1
        else:
            return 1

    def backup(self, value: float):
        """
        Propogates the value of some terminal state (the result of a rollout) up the tree 
        through the parent pointers.

        param value: The value to propogate up the tree.
        """
        self.total_value += value
        self.visitation_count += 1
        # Flips the value when calling backup because a positive reward for the current player is
        # a negative reward for the opposing player
        if self.parent is not None:
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

    # The value of the children is from the perspective of the opposing player, so negate it
    return average_value + node.prior * C * math.sqrt(visitation_ratio)

class MCTS:
    def __init__(self, env: AdversarialEnv, model, num_simulations=50):
        """
        param model: An object (theoretically some pytorch model) whose forward function takes in a 
            current board state and returns a probability distribution over the action space and predicted 
            value for the terminal state.
        param env: The environment of the game.
        param num_simulations: The number of simulations to go through in the run function.
        """
        self.env = env
        self.model = model
        self.num_simulations = num_simulations
        self.root_node = Node(None, None, 1, self.env.state)

    def step(self, action):
        for child in self.root_node.children:
            if child.action == action:
                child.parent = None
                child.action = None
                child.prior = 1
                self.root_node = child
                return

    def run(self):
        for i in range(self.num_simulations):
            # Find a leaf node
            curr_node = self.root_node
            while curr_node.children:
                ucb_scores = np.array([UCB1(node, curr_node.visitation_count) for node in curr_node.children])
                maximizing_child_idx = np.argmax(ucb_scores)
                curr_node = curr_node.children[maximizing_child_idx]

            # If the current node has never been visited, rollout from it and backup the value.
            # Otherwise expand the current node.
            if curr_node.visitation_count != 0:
                curr_node.expansion(self.model, self.env)
                if len(curr_node.children) > 0:
                    curr_node = curr_node.children[0]

            value = curr_node.rollout(self.model, self.env)
            curr_node.backup(value)
                

    def pi(self) -> ndarray:
        """
        Generate a probability distribution over the action space a move given the environment and the results of running MCTS. 

        return: A probability distribution over the action space. If the current state is terminal,
            an array of all zeros will be returned.
        """
        action_probs = np.zeros(self.env.action_space_shape)
        child_nodes = self.root_node.children
        for child_node in child_nodes:
            action = child_node.action
            action_probs[action] = child_node.visitation_count
        prob_sum = np.sum(action_probs)
        if prob_sum != 0:
            action_probs /= np.sum(action_probs)
        return action_probs

    def value(self) -> float:
        """
        Get the expected value from the current state. 

        return: A float indicating the expected reward from this state.
        """
        action_probs = np.zeros(self.env.action_space_shape)
        average_values = np.zeros(self.env.action_space_shape)
        child_nodes = self.root_node.children
        for child_node in child_nodes:
            action = child_node.action
            average_values[action] = child_node.total_value / child_node.visitation_count
            action_probs[action] = child_node.visitation_count
        prob_sum = np.sum(action_probs)
        if prob_sum != 0:
            action_probs /= np.sum(action_probs)
        return np.dot(average_values, action_probs)
