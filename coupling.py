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
from lispy import standard_env, eval, parse

class Workflow(object):
    """Workflow: Put things together"""
    def __init__(self, simulator: BaseSimulator, s: Type[State], a: Type[Action], test_policy=None):
        self.logger = logging.getLogger()
        self.simulator = simulator
        self.initial_state = s.get_initial_state()
        self.zero_action = a(0)
        self.type_state = s
        self.type_action = a
        self.test_policy = test_policy

        # self.game = Game()
        self.IRL = IRL(simulator, self.zero_action, gamma=0.9)
        input_dim = len(self.initial_state.feature_func())
        hidden_units = [128, 128, 32]
        batch_size = 16
        epochs = 4
        self.NN = NN(input_dim, hidden_units, batch_size, epochs)

        # scheme env
        self.global_env = standard_env()
        self.global_env.update({
            'state': self.type_state,
            'action': self.type_action,
            'repeat': self._repeat_game_with_policy,
            'train!': lambda traces: self._train_nn(traces, self._train_feature_weight_with_traces(traces)),
            'explore': Exploration,
            'random_policy': Exploration(ZeroPolicy(self.type_action), epsilon=1),
            'tree_search_policy': BaseTreeSearch(self.type_action, self.simulator, self.NN),
            'test_policy': self.test_policy,
        })

    def command(self, cmd: str):
        print('Executing:', cmd)
        eval(parse(cmd), env=self.global_env)

    def flow(self, _policy1, _policy2):
        """ this method is deprecated and preserved for debug use. """
        traces = self._repeat_game_with_policy(100, _policy1, _policy2)
        w = self._train_feature_weight_with_traces(traces)

        self.logger.info(w)
        print(w)
        print(w.shape)
        self._train_nn(traces, w)

        print('Testing 100 with trained policy and random policy')
        policy0 = BaseTreeSearch(self.type_action, self.simulator, self.NN)
        self._repeat_game_with_policy(20, policy0, _policy1)

        #print('Testing 100 with trained policy and trained policy')
        #policy0 = BaseTreeSearch(self.type_action, self.simulator, self.NN)
        # note the policy is stateless
        #self._repeat_game_with_policy(100, policy0, policy0)

        print('Testing 100 with trained policy and trained policy with exploration')
        policy01 = BaseTreeSearch(self.type_action, self.simulator, self.NN)
        policy01_exp = Exploration(policy01, epsilon=0.3)
        # note the policy is stateless
        self.traces = self._repeat_game_with_policy(20, policy01, policy01_exp)

        self._judge_policy(policy01, tag='v0.1', n=1)

        print('use replay of version 0.1 to train weight')
        w2 = self._train_feature_weight_with_traces(self.traces)
        print(w2)
        print(np.linalg.norm(w-w2))

        print('use replay of version 0.1 and weight to train nn')
        self._train_nn(self.traces, w2)

        print('Testing 100 with trained policy and trained policy with exploration')
        policy02 = BaseTreeSearch(self.type_action, self.simulator, self.NN)
        policy02_exp = Exploration(policy02, epsilon=0.3)
        self.traces = self._repeat_game_with_policy(100, policy02, policy02_exp)

        self._judge_policy(policy02, tag='v0.2', n=1)
        # hint: LoosedTrace(traces[2], wf.simulator).show(wf.IRL.coef, wf.NN)

    def _repeat_game_with_policy(self, n, policy1, policy2):
        self.game = Game(policy1, policy2)
        self.simulator.reset_cnt()
        total, win, tie, lose = 0, 0, 0, 0
        traces = []
        lose_idx = []
        ts = time.time()
        for e in range(n):
            self.game.reset()
            trace = self.game.start(self.simulator)
            reward = trace.final_state().reward()
            total += 1
            win += 1 if reward > 0 else 0
            tie += 1 if reward == 0 else 0
            lose += 1 if reward < 0 else 0
            if reward < 0:
                lose_idx.append(e)
            traces.append(trace)
            if n >= 5 and e % (n//5) == (n//5-1):
                print("[%-10s] %d%% - W/T/L: %d/%d/%d" % ('=' * (10 * (e + 1) // n), (100 * (e + 1) // n), win, tie, lose))

        print()
        t = time.time() - ts
        self.logger.info('W/T/L: %d/%d/%d - Time: %.1f s'%(win, tie, lose, t))
        print('W/T/L: %d/%d/%d - Time: %.1f s'%(win, tie, lose, t))
        print('Lose idx:', lose_idx)
        step_cnt, act_cnt = self.simulator.reset_cnt()
        print('step_cnt: %d, act_cnt: %d' % (step_cnt, act_cnt))
        return traces

    def _train_feature_weight_with_traces(self, traces):
        self.IRL.feed(traces) # the weight of each trace is same here
        # Maybe you can use cross validation to choose hyper param c
        param = self.IRL.train_w(hyper_param_c=1.0)
        return param

    def _train_nn(self, traces, weight):
        # get the training data of nn
        dd = [self.IRL._split_branch(trace, weight) for trace in traces]
        dd = [item for sublist in dd for item in sublist]
        data = [(sf, r) for s, sf, r in dd]
        size = len(data)

        X, y = [list(t) for t in zip(*data)]
        X = np.array(X)
        y = np.array(y).transpose()

        self.NN.build()
        self.NN.train(X, y)

        y_pred = np.ravel(self.NN.predict(X))
        self.y = y
        self.y_pred = y_pred

    def _judge_policy(self, policy_to_be_judged, tag='', n=100):
        if self.test_policy is None:
            return
        print('Judging %d with tag = %s' % (n, tag))
        self.judge_traces = self._repeat_game_with_policy(n, policy_to_be_judged, self.test_policy)
