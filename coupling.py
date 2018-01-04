#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File coupling.py created on 20:47 2018/1/1 

@author: Yichi Xiao
@version: 1.0
"""

import logging
from interfaces import *
from tree_search import *

class Workflow(object):
    """Workflow: Put things together"""
    def __init__(self, simulator: BaseSimulator, initial_state: State, zero_action: Action):
        self.logger = logging.getLogger()
        self.simulator = simulator
        self.initial_state = initial_state
        self.zero_action = zero_action
        self.type_state = type(initial_state)
        self.type_action = type(zero_action)

        # self.game = Game()
        self.IRL = IRL(simulator, zero_action)
        input_dim = len(initial_state.feature_func())
        hidden_units = [128, 128, 32]
        batch_size = 16
        epochs = 10
        self.NN = NN(input_dim, hidden_units, batch_size, epochs)

    def command(self):
        pass

    def flow(self, _policy1, _policy2):
        traces = self._repeat_game_with_policy(1000, _policy1, _policy2)
        w = self._train_feature_weight_with_traces(traces)
        for i, trace in enumerate(traces):
            print('Trace #%d' % i)
            LoosedTrace(trace, self.simulator).show(w)
        self.logger.info(w)
        print(w)
        print(w.shape)
        self._train_nn(traces, w)
        #for i, trace in enumerate(traces[0:100]):
        #    print('Trace #%d' % i)
        #    LoosedTrace(trace, self.simulator).show(w, self.NN)

        print('Testing 100 with trained policy and random policy')
        policy0 = BaseTreeSearch(self.type_action, self.simulator, self.NN)
        self._repeat_game_with_policy(100, policy0, _policy1)

        # LoosedTrace(traces[2], wf.simulator).show(wf.IRL.coef, wf.NN)

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
        print('W/T/L: %d/%d/%d'%(win,tie,lose))
        return traces

    def _train_feature_weight_with_traces(self, traces):
        self.IRL.feed(traces) # the weight of each trace is same here
        # Maybe you can use cross validation to choose hyper param c
        param = self.IRL.train_w(hyper_param_c=1.0)
        return param

    def _train_nn(self, traces, weight):
        # TODO what is the training data of nn
        dd = [self.IRL._split_branch(trace, weight) for trace in traces]
        dd = [item for sublist in dd for item in sublist]
        data = [(sf, r) for s, sf, r in dd]
        size = len(data)

        # X, y = zip(*data)
        X, y = [list(t) for t in zip(*data)]
        X = np.array(X)
        y = np.array(y).transpose()

        self.NN.build()
        self.NN.train(X, y)

        y_pred = np.ravel(self.NN.predict(X))
        self.y = y
        self.y_pred = y_pred

        #from sklearn.metrics import roc_auc_score
        #y_label = [0 if a < -0.5 else 1 if a < 0.5 else 2 for a in y]
        #auc = roc_auc_score(y_label, y_pred)  # label, score
        #print('AUC:', auc)

        import matplotlib.pyplot as plt
        plt.scatter(y, y_pred, c=None, marker='x', s=100)
        plt.axis('equal')
        plt.show()

        size = len(y)
        for idx in random.sample(range(size), k=10):
            st = dd[idx][0]
            print('[%04d] %s  label~: %+.4f nn~: %+.4f' % (idx, st, y[idx], y_pred[idx]))


