#!/usr/bin/python

from src.env.connect_four_env import ConnectFourEnv
from src.env.adversarial_env import AdversarialEnv, State
import random
from queue import LifoQueue
import sys
from copy import deepcopy
import numpy as np

# Throughout this file, ASP means adversarial search problem.
class Node:
    def __init__(self, parent=None, values=None, best_parent_values=None, state: State = None, parent_to_node_action=None, best_child_action=None, ply=0):
        self.parent = parent
        self.values = values
        self.best_parent_values = best_parent_values
        self.state = state
        self.parent_to_node_action = parent_to_node_action
        self.best_child_action = best_child_action
        self.ply = ply

def backup(node):
    parent = node.parent
    player = parent.state.current_player
    if parent.best_child_action is None or node.values[player] > parent.values[player]:
        parent.values = node.values
        parent.best_child_action = node.parent_to_node_action

def ply_backup(node):
    parent = node.parent
    player = parent.state.current_player
    if player == 1:
        if parent.best_child_action is None or node.values[0] > parent.values[0]:
            parent.values = node.values
            parent.best_child_action = node.parent_to_node_action
    if player == -1:
        if parent.best_child_action is None or node.values[0] < parent.values[0]:
            parent.values = node.values
            parent.best_child_action = node.parent_to_node_action


class StudentBot:
    def __init__(self, cutoff):
        self.cutoff = cutoff

    def decide(self, asp: AdversarialEnv):
        """
        Input: asp, a ConnectFourEnv
        Output: An int representing an action

        To get started, you can get the current
        state by calling asp.get_start_state()
        """
        return self.alpha_beta_cutoff(asp, self.cutoff, self.eval_func)

    def alpha_beta_cutoff(self, asp: AdversarialEnv, cutoff_ply, eval_func):
        """
        This function should:
        - search through the asp using alpha-beta pruning
        - cut off the search after cutoff_ply moves have been made.

        Inputs:
                asp - an AdversarialSearchProblem
                cutoff_ply- an Integer that determines when to cutoff the search
                        and use eval_func.
                        For example, when cutoff_ply = 1, use eval_func to evaluate
                        states that result from your first move. When cutoff_ply = 2, use
                        eval_func to evaluate states that result from your opponent's
                        first move. When cutoff_ply = 3 use eval_func to evaluate the
                        states that result from your second move.
                        You may assume that cutoff_ply > 0.
                eval_func - a function that takes in a GameState and outputs
                        a real number indicating how good that state is for the
                        player who is using alpha_beta_cutoff to choose their action.
                        You do not need to implement this function, as it should be provided by
                        whomever is calling alpha_beta_cutoff, however you are welcome to write
                        evaluation functions to test your implemention. The eval_func we provide
            does not handle terminal states, so evaluate terminal states the
            same way you evaluated them in the previous algorithms.

        Output: an action(an element of asp.get_available_actions(asp.get_start_state()))
        """
        start_state = State(asp.state.board.copy(), asp.state.current_player)
        start_node = Node(state=start_state)
        node_stack = LifoQueue()
        node_stack.put(start_node)
        while not node_stack.empty():
            node = node_stack.get()
            if node.state != start_state and node.parent.values is not None and node.parent.best_parent_values is not None:
                parent = node.parent
                parent_player = parent.state.current_player
                if parent_player == 1:
                    if parent.values[0] >= parent.best_parent_values[0]:
                        continue
                if parent_player == -1:
                    if parent.values[0] <= parent.best_parent_values[0]:
                        continue
            if node.state is None:
                node.state = asp.perform_action_on_state(node.parent.state, node.parent_to_node_action)
            if asp.is_terminal_state(node.state):
                values = asp.evaluate_state(node.state)
                node.values = values
            elif node.ply == cutoff_ply:
                node.values = eval_func(node.state)
            if node.values is not None:
                if node.state != start_state:
                    ply_backup(node)
            else:
                if node.state != start_state and node.parent.values is not None:
                    node.best_parent_values = node.parent.values
                node_stack.put(node)
                actions = asp.get_available_actions(node.state)
                actions_list = []
                for action in actions:
                    actions_list += [action]
                for action in reversed(actions_list):
                    next_node = Node(parent=node, parent_to_node_action=action, ply=node.ply+1)
                    node_stack.put(next_node)
        action = np.zeros(asp.action_space_shape)
        action[start_node.best_child_action] = 1
        return action

    def eval_func(self, state):
        random_score = random.uniform(0.4, 0.6)
        return random_score, 1 - random_score
