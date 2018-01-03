#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File coupling.py created on 20:47 2018/1/1 

@author: Yichi Xiao
@version: 1.0
"""

import logging
from interfaces import *

class Workflow(object):
    """Workflow: Put things together"""
    def __init__(self, simulator, initial_state, zero_action):
        self.logger = logging.getLogger()
        self.simulator = simulator
        self.initial_state = initial_state
        self.zero_action = zero_action
        self.type_state = type(initial_state)
        self.type_action = type(zero_action)

        # self.game = Game()
        self.IRL = IRL(simulator, zero_action)
        input_dim = len(initial_state)
        hidden_units = [32, 32]
        batch_size = 16
        epochs = 10
        self.NN = NN(input_dim, hidden_units, batch_size, epochs)

    def command(self):
        pass

    def _repeat_game_with_policy(self, n, policy1, policy2):
        self.game = Game(policy1, policy2)
        total, win, tie, lose = 0, 0, 0, 0
        traces = []
        for i in range(n):
            self.game.reset()
            trace = self.game.start(self.simulator)
            reward = trace.final_state().reward()
            total += 1
            win += 1 if reward > 0 else 0
            tie += 1 if reward == 0 else 0
            lose += 1 if reward < 0 else 0
            traces.append(trace)

        self.logger.info('W/T/L: %d/%d/%d'%(win,tie,lose))
        return traces

    def _train_feature_weight_with_traces(self, traces):
        self.IRL.feed(traces) # the weight of each trace is same here
        # Maybe you can use cross validation to choose hyper param c
        param = self.IRL.train_w(hyper_param_c=1.0)
        return param

    def _train_nn(self, traces, weight):
        # TODO what is the training data of nn
        dd = [self.IRL._split_branch(trace, weight) for trace in traces]
        data = reduce(lambda x,y: x+y, dd, [])
        size = len(data)

        X, y = zip(*data)
        # X, y = [list(t) for t in zip(*data)]

        self.NN.build()
        self.NN.train(X, y)
