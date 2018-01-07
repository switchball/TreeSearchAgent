#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File test_rps.py created on 9:51 2018/1/5 

@author: Yichi Xiao
@version: 1.0
"""
from unittest import TestCase
from interfaces import *
from game_pong import *


class TestPong(TestCase):
    def setUp(self):
        self.game = Game(Exploration(ZeroPolicy(PongAction), 0), Exploration(ZeroPolicy(PongAction), 0))
        self.simulator = PongSimulator()
        self.IRL = IRL(self.simulator, PongAction(0), gamma=0.9)

        input_dim = len(PongState.get_initial_state().feature_func())
        hidden_units = [64, 32]
        batch_size = 16
        epochs = 6
        self.NN = NN(input_dim, hidden_units, batch_size, epochs)

    def tearDown(self):
        pass

    def test_feature_sparsity(self):
        # random generate 100 samples
        cnt = 0
        size = len(PongState.get_initial_state().feature_func())
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
