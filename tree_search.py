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
                # TODO
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