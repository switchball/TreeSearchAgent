#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File test_rps.py created on 9:51 2018/1/5 

@author: Yichi Xiao
@version: 1.0
"""
from unittest import TestCase
from interfaces import *
from game_rps import *


class TestRPS(TestCase):
    def setUp(self):
        self.game = Game(RPSRandomPolicy(), RPSRandomPolicy())
        self.simulator = RPSSimulator()
        self.IRL = IRL(self.simulator, RPSAction(0), gamma=0.9)

        input_dim = len(RPSState.get_initial_state().feature_func())
        hidden_units = [128, 128, 32]
        batch_size = 16
        epochs = 6
        self.NN = NN(input_dim, hidden_units, batch_size, epochs)

    def tearDown(self):
        pass

    def test_feature_sparsity(self):
        # random generate 100 samples
        cnt = 0
        size = len(RPSState.get_initial_state().feature_func())
        sparsity = np.zeros(size)
        examples = [None] * size  # type: List[State]
        for i in range(200):
            self.game.reset()
            trace = self.game.start(self.simulator)
            reward = trace.final_state().reward()
            for state in trace.states:
                fea = state.feature_func(view=1)
                sparsity += fea != 0
                for k in range(size):
                    if examples[k] is None and fea[k] != 0:
                        examples[k] = state
                        break

                cnt += 1

        zero_list = []
        for i in range(size):
            print('Feature %02d: %02.2f = %d / %d\t%s' % (i, 100 * sparsity[i] / cnt, sparsity[i], cnt, examples[i]))
            if sparsity[i] == 0:
                zero_list.append(i)
        if len(zero_list) > 0:
            self.fail('feature %s seems to have constant zero value!' % zero_list)

    def test_feature_sign(self):
        traces = []
        for _ in range(100):
            self.game.reset()
            trace = self.game.start(self.simulator)
            traces.append(trace)
        self.IRL.feed(traces)
        w = self.IRL.train_w(hyper_param_c=1.0)

        st0 = RPSState.get_initial_state()
        self.assertAlmostEqual(np.dot(st0.feature_func(view=1), w), 0, delta=0.1)
        self.assertAlmostEqual(np.dot(st0.feature_func(view=2), w), 0, delta=0.1)

        st1 = self.simulator.next(st0, RPSAction(1), RPSAction(0))
        self.assertGreater(np.dot(st1.feature_func(view=1), w), 0, 'For player1, R------ > 0')
        self.assertLess(np.dot(st1.feature_func(view=2), w), 0, 'For player2, R------ < 0')

        st2 = self.simulator.next(st0, RPSAction(0), RPSAction(1))
        self.assertLess(np.dot(st2.feature_func(view=1), w), 0, 'For player1, ------r < 0')
        self.assertGreater(np.dot(st2.feature_func(view=2), w), 0, 'For player2, ------r > 0')

        st3 = self.simulator.next(st0, RPSAction(1), RPSAction(3))
        self.assertLessEqual(np.dot(st3.feature_func(view=1), w), 0, 'For player1, R-----p < 0')
        self.assertGreaterEqual(np.dot(st3.feature_func(view=2), w), 0, 'For player2, R-----p > 0')

    def test_nn_feature_sign(self):
        traces = []
        for _ in range(100):
            self.game.reset()
            trace = self.game.start(self.simulator)
            traces.append(trace)
        self.IRL.feed(traces)
        w = self.IRL.train_w(hyper_param_c=1.0)

        dd = [self.IRL._split_branch(trace, w) for trace in traces]
        dd = [item for sublist in dd for item in sublist]
        data = [(sf, r) for s, sf, r in dd]

        X, y = [list(t) for t in zip(*data)]
        X = np.array(X)
        y = np.array(y).transpose()

        self.NN.build()
        self.NN.train(X, y)

        st0 = RPSState.get_initial_state()
        self.assertAlmostEqual(self.NN.predict_one(st0.feature_func(view=1)), 0, delta=0.1)
        self.assertAlmostEqual(self.NN.predict_one(st0.feature_func(view=2)), 0, delta=0.1)

        st1 = self.simulator.next(st0, RPSAction(1), RPSAction(0))
        self.assertGreater(self.NN.predict_one(st1.feature_func(view=1)), 0, 'For player1, R------ > 0')
        self.assertLess   (self.NN.predict_one(st1.feature_func(view=2)), 0, 'For player2, R------ < 0')

        st2 = self.simulator.next(st0, RPSAction(0), RPSAction(1))
        self.assertLess   (self.NN.predict_one(st2.feature_func(view=1)), 0, 'For player1, ------r < 0')
        self.assertGreater(self.NN.predict_one(st2.feature_func(view=2)), 0, 'For player2, ------r > 0')

        st3 = self.simulator.next(st0, RPSAction(1), RPSAction(3))
        self.assertLess   (self.NN.predict_one(st3.feature_func(view=1)), 0, 'For player1, R-----p < 0')
        self.assertGreater(self.NN.predict_one(st3.feature_func(view=2)), 0, 'For player2, R-----p > 0')