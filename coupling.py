#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File coupling.py created on 20:47 2018/1/1 

@author: Yichi Xiao
@version: 1.0
"""

import time
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
        self.IRL = IRL(simulator, zero_action, gamma=0.9)
        input_dim = len(initial_state.feature_func())
        hidden_units = [128, 128, 32]
        batch_size = 16
        epochs = 6
        self.NN = NN(input_dim, hidden_units, batch_size, epochs)

    def command(self):
        pass

    def flow(self, _policy1, _policy2):
        traces = self._repeat_game_with_policy(100, _policy1, _policy2)
        w = self._train_feature_weight_with_traces(traces)
        # for i, trace in enumerate(traces):
            # print('Trace #%d' % i)
            # LoosedTrace(trace, self.simulator).show(w)
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

        #print('Testing 100 with trained policy and trained policy')
        #policy0 = BaseTreeSearch(self.type_action, self.simulator, self.NN)
        # note the policy is stateless
        #self._repeat_game_with_policy(100, policy0, policy0)

        print('Testing 100 with trained policy and trained policy with exploration')
        policy0 = BaseTreeSearch(self.type_action, self.simulator, self.NN)
        policy_exp = Exploration(policy0, epsilon=0.3)
        # note the policy is stateless
        self.traces = self._repeat_game_with_policy(100, policy0, policy_exp)

        # LoosedTrace(traces[2], wf.simulator).show(wf.IRL.coef, wf.NN)

    def _repeat_game_with_policy(self, n, policy1, policy2):
        self.game = Game(policy1, policy2)
        self.simulator.reset_cnt()
        total, win, tie, lose = 0, 0, 0, 0
        traces = []
        ts = time.time()
        for i in range(n):
            self.game.reset()
            trace = self.game.start(self.simulator)
            reward = trace.final_state().reward()
            total += 1
            win += 1 if reward > 0 else 0
            tie += 1 if reward == 0 else 0
            lose += 1 if reward < 0 else 0
            traces.append(trace)
            if n % (n//25) == 0:
                print('>', end='')

        print()
        t = time.time() - ts
        self.logger.info('W/T/L: %d/%d/%d - Time: %.1f s'%(win, tie, lose, t))
        print('W/T/L: %d/%d/%d - Time: %.1f s'%(win, tie, lose, t))
        step_cnt, act_cnt = self.simulator.reset_cnt()
        print('step_cnt: %d, act_cnt: %d' % (step_cnt, act_cnt))
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

        import matplotlib.cm
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        # Calculate the point density
        xy = np.vstack([y, y_pred])
        z0 = gaussian_kde(xy)(xy)

        # Sort the points by density, so that the densest points are plotted last
        idx = z0.argsort()
        x0, y0, z0 = y[idx], y_pred[idx], z0[idx]

        fig, ax = plt.subplots()
        ax.scatter(x0, y0, c=z0, s=50, edgecolor='')
        plt.axis('equal')
        plt.show()

        size = len(y)
        distances = np.array([abs(a-b) for a,b in zip(y, y_pred)])

        idxes = distances.argsort()

        for idx in idxes[-30:]:
            st = dd[idx][0]
            print('[%04d] %s  label~: %+.4f nn~: %+.4f' % (idx, st, y[idx], y_pred[idx]))

        for idx in range(size):
            st = dd[idx][0]
            if st.flag != 0:
                print('[%04d] %s  label~: %+.4f nn~: %+.4f' % (idx, st, y[idx], y_pred[idx]))

