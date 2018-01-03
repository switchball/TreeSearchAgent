#!/user/bin/env python
# -*- coding: utf-8 -*-
#
# The interfaces of ???

from keras.models import Sequential
from keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
import numpy as np
import random
from functools import reduce
from typing import List, Optional


class State(object):
    """State: Abstract class for game state representation"""
    def __init__(self, s):
        self.s = s

    @staticmethod
    def get_initial_state():
        raise NotImplementedError

    def feature_for_player1(self) -> np.ndarray:
        raise NotImplementedError

    def feature_for_player2(self) -> np.ndarray:
        raise NotImplementedError

    def feature_func(self, view: int = 1) -> np.ndarray:
        v1 = self.feature_for_player1()
        v2 = self.feature_for_player2()
        if view == 1:
            return v1 - v2
        elif view == 2:
            return v2 - v1
        else:
            assert 'view should either be 1 or 2!'

    def stableQ(self) -> bool:
        raise NotImplementedError

    def terminateQ(self) -> bool:
        raise NotImplementedError

    def reward(self) -> int:
        raise NotImplementedError

    def dump_to_json(self):
        pass

    def load_from_json(self, js):
        pass


class Action(object):
    """Action: Abstract class for game action representation"""
    def __init__(self, a):
        self.a = a

    @staticmethod
    def get_action_spaces():
        pass

    def zeroQ(self):
        return self.a == 0


class Policy(object):
    """Policy: strategy mapping from State to Action"""
    def __init__(self):
        pass

    def action(self, state):
        raise NotImplementedError


class ZeroPolicy(Policy):
    """ZeroPolicy: a policy which always returns zero action"""
    def __init__(self, zero_action=Action(0)):
        super(ZeroPolicy, self).__init__()
        self.zero_action = zero_action

    def action(self, state):
        return self.zero_action


class BaseSimulator(object):
    """BaseSimulator: Provide n-step simulation of the state"""
    def __init__(self):
        pass

    def initial_state(self):
        # Note: should return State.get_initial_state()
        raise NotImplementedError

    def _step_env(self, state, copy=True):
        raise NotImplementedError

    def _step_act(self, state, a1, a2, copy=False):
        return self.next2(self.next1(state, a1, copy), a2, copy)

    def next1(self, state, a1: Action, copy=False):
        raise NotImplementedError

    def next2(self, state, a2: Action, copy=False):
        raise NotImplementedError

    def next(self, state, a1, a2):
        """ First environment step, then action step """
        s = self._step_env(state, copy=True)
        return self._step_act(s, a1, a2, copy=False)

    def next_empty_n(self, state, n):
        s = state
        ss = []
        for i in range(n):
            s = self._step_env(s)
            ss.append(s)
        return ss

    def next_action_n(self, state, actions, n, max_t=100):
        a1, a2 = actions
        s = state
        ss = []
        # trans = lambda s: self.next2(self.next1(state, a1), a2)
        for i in range(min(n, max_t)):
            s = self.next(s, a1, a2)
            ss.append(s)
            if s.stableQ():
                break
        return ss



class Game(object):
    """Game: Provide basic two-player zero-sum logic"""
    def __init__(self, policy1, policy2):
        self.policy1 = policy1
        self.policy2 = policy2

    def reset(self):
        self.t = 0
        self.max_t = 15
        return self

    def start(self, simulator: BaseSimulator):
        """
        :type simulator: BaseSimulator
        :rtype: Trace
        :return: the trace generated in the game play
        """
        st = simulator.initial_state() # type: State
        trace = Trace(st)
        t = 0
        while t < self.max_t and not st.terminateQ():
            action1 = self.policy1.action(st)
            action2 = self.policy2.action(st)
            s = simulator._step_env(st)
            s = simulator._step_act(s, action1, action2)
            # add s0, a1, a2, s to Trace
            trace.append(s, action1, action2)
            st = s
            t += 1

        # How to get the reward? trace.final_state().reward()
        self.trace = trace
        return trace


class IRL(object):
    """IRL: Inverse Reinforcement Learning Module"""
    def __init__(self, simulator: BaseSimulator, zero_action, gamma=0.99):
        self.gamma = gamma
        self.zero_action = zero_action
        self.bs = simulator

    def feed(self, traces):
        # Prepare Data
        dd = [self._split_advantage(trace) for trace in traces]
        self.data = reduce(lambda x,y: x+y, dd, [])
        size = len(self.data)

        # Random Flip Some Data
        # class 0 means good (x1), class 1 means bad (x-1)
        rs = random.choices([0, 1], k=size)
        self.X = [d * (1-2*r) for d, r in zip(self.data, rs)]
        self.y = rs
        return self.X, self.y


    def train_w(self, hyper_param_c=1.0):
        # TODO IRL Algorithm needed

        # Train with linear svm (with no bias)
        model = LinearSVC(random_state=0, fit_intercept=False)
        model.C = hyper_param_c
        model.fit(self.X, self.y)

        self.coef = model.coef_
        return model.coef_

    def _split_advantage(self, trace_):
        ret = []
        winner = 1 if trace_.final_state().reward() >= 0 else 2
        split_arr = trace_.split()
        for trace in split_arr:
            s0 = trace.states[0]
            discounted_r = [pow(self.gamma, k) * s.feature_func(view=winner)
                             * ((1.0/(1-self.gamma) if k == len(trace.states) - 1 else 1))
                             for k, s in enumerate(trace.states)]
            mu_star = sum(discounted_r)
            #route_states = self.bs.next_action_n(s0, trace.actions[0], 100)
            zero_act = (self.zero_action, trace.actions[0][1]) if winner == 1 else \
                       (trace.actions[0][0], self.zero_action)
            route_states = self.bs.next_action_n(s0, zero_act, 100)
            discounted_r = [pow(self.gamma, k) * s.feature_func(view=winner)
                             * ((1.0/(1-self.gamma) if k == len(route_states) - 1 else 1))
                             for k, s in enumerate(route_states)]
            mu_bad = sum(discounted_r)
            ret.append(mu_star - mu_bad)
        # => [(s0, a1, s1, dis_sum_features, s_final)]
        return ret

    def _split_branch(self, trace_, w_):
        ret = []
        winner = 1 if trace_.final_state().reward() >= 0 else 2
        split_arr = trace_.split()
        for trace in split_arr:
            # sample (s1, r_s1) from trace
            s1 = trace.states[1]
            r_s1 = [pow(self.gamma, k) * np.dot(s.feature_func(view=winner), w_)
                     * ((1.0/(1-self.gamma) if k == len(trace.states[1:]) - 1 else 1))
                     for k, s in enumerate(trace.states[1:])]

            # sample (s0, r_s0) from simulation (should use TODO pure zero_act?
            s0 = trace.states[0]
            zero_act = (self.zero_action, trace.actions[0][1]) if winner == 1 else \
                (trace.actions[0][0], self.zero_action)
            route_states = self.bs.next_action_n(s0, zero_act, 100)
            r_s0 = [pow(self.gamma, k) * np.dot(s.feature_func(view=winner), w_)
                     * ((1.0/(1-self.gamma) if k == len(route_states) - 1 else 1))
                     for k, s in enumerate(route_states)]

            ret.append((s1, r_s1))
            ret.append((s0, r_s0))

        return ret


class NN(object):
    """NN: Neural Network for learning value function of state"""
    def __init__(self, input_dim, hidden_units, batch_size, epochs):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = 1
        self.model = self.build()
        self.batch_size = batch_size
        self.epochs = epochs

    def build(self):
        model = Sequential()
        hs = self.hidden_units  # just name alias
        assert len(hs) > 0
        model.add(Dense(hs[0], input_dim=self.input_dim, activation='relu'))
        for n in hs[1:]:
            model.add(Dense(n, activation='relu')) # hidden layers
        model.add(Dense(self.output_dim)) # output value with no activation
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def train(self, data, labels):
        self.model.fit(data, labels, validation_split=0.2, epochs=self.epochs,
                        batch_size=self.batch_size)

    def train_kfold(self, X, Y):
        from sklearn.model_selection import StratifiedKFold
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cvscores = []
        for train, test in kfold.split(X, Y):
            # create model
            model = self.build()
            # Fit the model
            model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
            # evaluate the model
            scores = model.evaluate(X[test], Y[test], verbose=0)
            print("%s: %.2f%%" % (model.metrics_names[0], scores[0]))
            cvscores.append(scores[0])
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    def predict(self, data):
        return self.model.predict(data)


class Trace(object):
    """Trace: Trajectory with shape like (s_0, a_1, s_1, ..., a_n, s_n) """
    def __init__(self, s0, state_list=None, action_list=None):
        self.states = [s0]
        self.actions = [(Action(0),Action(0))] # placeholder for action[0]
        self.step = 0
        if state_list is not None and action_list is not None:
            assert len(state_list) == len(action_list)
            self.states += state_list
            self.actions += action_list
            self.step += len(state_list)

    @classmethod
    def from_trace(cls, traces, start_idx, end_idx):
        states = traces.states[(start_idx-1):(end_idx+1)]
        actions = traces.actions[start_idx:(end_idx+1)]
        return cls(s0=states[0], state_list=states[1:], action_list=actions)

    def final_state(self) -> State:
        return self.states[-1]

    def show(self): # IO
        for k, s in enumerate(self.states):
            print('%03d: %s' % (k, str(s)))

    def append(self, s, a1, a2):
        self.states.append(s)
        self.actions.append((a1, a2))
        self.step += 1

    def split(self):
        isP1Winner = True if self.final_state().reward() >= 0 else False
        trace_arr = []
        t_start, t_effect = None, None
        for t in range(1,self.step):
            s = self.states[t]
            if isP1Winner: # 利用转置性质绕过分割逻辑，需要与feature_func同步使用
                a1, a2 = self.actions[t]
            else:
                a2, a1 = self.actions[t]
            if s.stableQ:
                if t_effect is None:
                    # fail, reset and continue
                    t_start, t_effect = None, None
                    continue
                else:
                    # range from [t_effect, self.step]
                    split_trace = Trace.from_trace(self, t_effect, self.step)
                    trace_arr.append(split_trace)
                    # reset and continue
                    t_start, t_effect = None, None
                    continue
            else:
                t_start = self.step
            if a1.zeroQ() and a2.zeroQ():
                pass
            if a1.zeroQ() and not a2.zeroQ():
                t_start, t_effect = t_effect, None
                if t_start is None:
                    t_start = self.step
            if not a1.zeroQ() and a2.zeroQ():
                if t_effect is None:
                    t_effect = self.step
                else:
                    t_start, t_effect = t_effect, self.step
            if not a1.zeroQ() and not a2.zeroQ():
                t_start, t_effect = self.step, self.step

        return trace_arr

