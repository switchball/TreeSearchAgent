#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File tree_search.py created on 21:21 2018/1/4 

@author: Yichi Xiao
@version: 1.0
"""

from interfaces import *


class BaseTreeSearch(Policy):
    """BaseTreeSearch: Base class for tree search"""
    def __init__(self, action_type, simulator : BaseSimulator, nn : NN):
        super(BaseTreeSearch, self).__init__(action_type)
        self.simulator = simulator
        self.NN = nn

    def action(self, state, player=1):
        """
        In the implementation, the agent look one step ahead
        and use mini-max on opponent's behaviour
        """
        s_ = self.simulator._step_env(state, copy=True)
        action_space = self.action_type.get_action_spaces()
        size = len(action_space)
        state_features = [None] * (size ** 2)
        idx = 0
        for a_me in action_space:
            for a_op in action_space:
                if player == 1:
                    st = self.simulator._step_act(s_, a_me, a_op, copy=True)
                elif player == 2:
                    st = self.simulator._step_act(s_, a_op, a_me, copy=True)
                else:
                    st = None  # player should be 1 or 2
                state_features[idx] = st.feature_func(view=player)
                idx += 1
        vs = self.NN.predict(np.array(state_features))
        vs = vs.reshape((size, size))

        # minimize on opponent's choice, maximize on player's choice
        k = np.argmax(np.min(vs, axis=1))
        return action_space[k]


class MinimaxSearch(Policy):
    def __init__(self, action_type, simulator: BaseSimulator, nn: NN, ignore_opponent=True):
        super(MinimaxSearch, self).__init__(action_type)
        self.simulator = simulator
        self.NN = nn
        self.action_space = self.action_type.get_action_spaces()
        self.ignore_opponent = ignore_opponent

    def action(self, state, player=1):
        """
        In the implementation, the agent will use minimax search on given depth
        """
        value = state.score_by(self.NN, view=player)
        if value >= -20:
            v, idx = self.minimax(state, 1*2, True, player)
        elif value >= -50:
            v, idx = self.minimax(state, 2*2, True, player)
        elif value >= -200:
            v, idx = self.minimax(state, 3*2, True, player)
        else:
            v, idx = self.minimax(state, 4*2, True, player)
        return self.action_space[idx]

    def minimax(self, state, depth, maximizing, player):
        idx = -1
        if depth == 0 or state.terminateQ():
            return state.score_by(self.NN, view=player), idx
        if maximizing:  # maximizing player
            v = float('-inf')
            s_ = self.simulator._step_env(state, copy=True)
            if player == 1:
                children = [self.simulator.next1(s_, a, copy=True) for a in self.action_space]
            else:
                children = [self.simulator.next2(s_, a, copy=True) for a in self.action_space]
            for k, child in enumerate(children):
                _v, _idx = self.minimax(child, depth - 1, False, player)
                if _v > v:
                    idx = k
                    v = _v
                # v = max(v, _v)
                # alpha = max(alpha, v)
            return v, idx
        else:           # minimizing player
            v = float('inf')
            s_ = state  # do not need to call simulator._step_env again !!  # name alias
            if self.ignore_opponent:
                children = [s_]  # as if call next with zero action
            elif player == 1:
                children = [self.simulator.next2(s_, a, copy=True) for a in self.action_space]
            else:
                children = [self.simulator.next1(s_, a, copy=True) for a in self.action_space]
            for k, child in enumerate(children):
                _v, _idx = self.minimax(child, depth - 1, True, player)
                if _v < v:
                    idx = k
                    v = _v
                # v = min(v, _v)
                # beta = min(beta, v)
            return v, idx
