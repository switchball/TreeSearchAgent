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
        and ignore the opponent's behaviour
        """
        vs = []
        s_ = self.simulator._step_env(state, copy=True)
        action_space = self.action_type.get_action_spaces()
        for a in action_space:
            if player == 1:
                st = self.simulator._step_act(s_, a, Action(0), copy=True)
            elif player == 2:
                st = self.simulator._step_act(s_, Action(0), a, copy=True)
            else:
                st = None  # player should be 1 or 2
            v = self.NN.predict_one(st.feature_func(view=player))
            vs.append(v)

        idx = np.argmax(vs)
        return action_space[idx]