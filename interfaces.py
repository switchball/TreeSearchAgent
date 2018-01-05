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
from itertools import repeat, chain
from functools import reduce
from typing import List, Tuple, Type, Optional


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

    def visual(self, w=None):
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
    def __init__(self, action_type: Type[Action]):
        self.action_type = action_type

    def action(self, state, player=1) -> Action:
        raise NotImplementedError

    def reset(self):
        pass


class ZeroPolicy(Policy):
    """ZeroPolicy: a policy which always returns zero action"""
    def __init__(self, action_type):
        super(ZeroPolicy, self).__init__(action_type)
        self.zero_action = self.action_type(0)

    def action(self, state, player=1):
        return self.zero_action


class Exploration(Policy):
    def __init__(self, base_policy: Policy, epsilon=0.05):
        super(Exploration, self).__init__(base_policy.action_type)
        self.base_policy = base_policy
        self.epsilon = epsilon
        self.action_space = self.action_type.get_action_spaces()

    def action(self, state, player=1):
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        else:
            return self.base_policy.action(state, player)


class BaseSimulator(object):
    """BaseSimulator: Provide n-step simulation of the state"""
    def __init__(self):
        pass

    def initial_state(self):
        # Note: should return State.get_initial_state()
        raise NotImplementedError

    def _step_env(self, state: State, copy=True):
        raise NotImplementedError

    def _step_act(self, state: State, a1: Action, a2: Action, copy=False):
        return self.next2(self.next1(state, a1, copy), a2, copy)

    def next1(self, state: State, a1: Action, copy=False) -> State:
        raise NotImplementedError

    def next2(self, state: State, a2: Action, copy=False) -> State:
        raise NotImplementedError

    def next(self, state, a1, a2):
        """ First environment step, then action step """
        s = self._step_env(state, copy=True)
        return self._step_act(s, a1, a2, copy=False)

    def next_empty_some(self, state, f=lambda s: s.stableQ() or s.terminateQ(), max_t=100):
        s = state
        ss = []
        for i in range(max_t):
            s = self._step_env(s)
            ss.append(s)
            if f(s):
                break
        return ss

    def next_action_some(self, state, actions, f=lambda s: s.stableQ() or s.terminateQ(), max_t=100):
        a1, a2 = actions
        s = state
        ss = []
        for i in range(max_t):
            s = self._step_env(s)
            if i == 0:  # only execute action at first time step
                s = self._step_act(s, a1, a2)
            ss.append(s)
            if f(s):
                break
        return ss


class Game(object):
    """Game: Provide basic two-player zero-sum logic"""
    def __init__(self, policy1: Policy, policy2: Policy):
        self.policy1 = policy1
        self.policy2 = policy2

    def reset(self):
        self.t = 0
        self.max_t = 25
        self.policy1.reset()
        self.policy2.reset()
        return self

    def start(self, simulator: BaseSimulator):
        """
        :type simulator: BaseSimulator
        :rtype: Trace
        :return: the trace generated in the game play
        """
        st = simulator.initial_state()  # type: State
        trace = Trace(st)
        t = 0
        while t < self.max_t and not st.terminateQ():
            action1 = self.policy1.action(st, player=1)
            action2 = self.policy2.action(st, player=2)
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
    def __init__(self, simulator: BaseSimulator, zero_action, gamma=0.9):
        self.gamma = gamma
        self.zero_action = zero_action
        self.bs = simulator

    def feed(self, traces):
        # Prepare Data
        dd = [self._split_advantage(trace) for trace in traces]
        self.data = [item for sublist in dd for item in sublist]

        size = len(self.data)

        assert size > 2, 'input size too small'

        # Random Flip Some Data
        # class 0 means good (x-1), class 1 means bad (x+1)
        rs = list(chain(repeat(0, size // 2), repeat(1, size - size // 2)))
        random.shuffle(rs)
        print(rs)
        self.X = [d * (2*r-1) for d, r in zip(self.data, rs)]
        self.y = rs
        return self.X, self.y

    def train_w(self, hyper_param_c=1.0):
        # TODO IRL Algorithm needed

        # Train with linear svm (with no bias)
        model = LinearSVC(random_state=0, fit_intercept=False)
        model.C = hyper_param_c
        model.fit(self.X, self.y)

        self.coef = np.ravel(model.coef_)
        return self.coef

    def _split_advantage(self, trace) -> List[np.ndarray]:
        ret = []
        lt = LoosedTrace(trace, self.bs)
        for winner in (1, 2):  # enumerate the winner in player1 and player2
            split_trace_arr = lt.split(view=winner)
            for good_trace, weak_trace in split_trace_arr:
                good_dis_fea = [pow(self.gamma, k) * s.feature_func(view=winner)
                                * (1.0/(1-self.gamma) if k == len(good_trace) - 1 else 1)
                                for k, s in enumerate(good_trace)]
                weak_dis_fea = [pow(self.gamma, k) * s.feature_func(view=winner)
                                * (1.0 / (1 - self.gamma) if k == len(weak_trace) - 1 else 1)
                                for k, s in enumerate(weak_trace)]
                mu_good = sum(good_dis_fea)
                mu_weak = sum(weak_dis_fea)
                ret.append(mu_good - mu_weak)
            # the final state is important
            # final_reward = lt.final_state().reward()
            # if final_reward > 0:  # it should be better than any given state
            #     st = lt.states[random.choice(range(lt.step))]


        return ret

    def _split_branch(self, trace, w) -> List[Tuple[State, np.ndarray, float]]:
        ret = []
        lt = LoosedTrace(trace, self.bs)
        for winner in (1, 2):  # enumerate the winner in player1 and player2
            split_trace_arr = lt.split(view=winner)
            for good_trace, weak_trace in split_trace_arr:
                good_dis_fea = [pow(self.gamma, k) * s.feature_func(view=winner)
                                * (1.0/(1-self.gamma) if k == len(good_trace) - 1 else 1)
                                for k, s in enumerate(good_trace)]
                weak_dis_fea = [pow(self.gamma, k) * s.feature_func(view=winner)
                                * (1.0 / (1 - self.gamma) if k == len(weak_trace) - 1 else 1)
                                for k, s in enumerate(weak_trace)]
                mu_good = sum(good_dis_fea)
                mu_weak = sum(weak_dis_fea)
                r_good = np.dot(mu_good, w)
                r_weak = np.dot(mu_weak, w)
                ret.append((good_trace[0], good_trace[0].feature_func(view=winner), r_good))
                ret.append((weak_trace[0], weak_trace[0].feature_func(view=winner), r_weak))
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
            model.add(Dense(n, activation='relu'))  # hidden layers
        model.add(Dense(self.output_dim))  # output value with no activation
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

    def predict_one(self, input: np.ndarray) -> float:
        return self.predict(input.reshape(1, -1))[0, 0]


class Trace(object):
    """Trace: Trajectory with shape like (s_0, a_1, s_1, ..., a_n, s_n) """
    def __init__(self, s0, state_list=None, action_list=None):
        self.states = [s0]  # type: List[State]
        self.actions = [(Action(0), Action(0))]  # placeholder for action[0]
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

    def show(self, w=None):  # IO
        for k, s in enumerate(self.states):
            print('%03d: %s' % (k, s.visual(w)))

    def append(self, s, a1, a2):
        self.states.append(s)
        self.actions.append((a1, a2))
        self.step += 1

    def split(self):
        raise NotImplementedError('Please use LoosedTrace(trace).split() instead.')


class LoosedTrace(Trace):
    """LoosedTrace: Trace with simulated zero-action branch"""
    def __init__(self, trace, simulator):
        super(LoosedTrace, self).__init__(trace.states[0], trace.states[1:], trace.actions[1:])
        self.zero_1 = [None] * self.step  # simulated zero-act for player 1
        self.zero_2 = [None] * self.step  # simulated zero-act for player 2
        self.real_act = [None] * self.step  # (mixed simulated) real-act
        self.side_chain = []  # contains simulated trace (but with no actions)
        self._loose(simulator)

    def _loose(self, simulator: BaseSimulator):
        snap_idx = -1                              # terminal state's idx = -1
        snap_reward = self.final_state().reward()  # snapshot reward is final state's reward
        for t in reversed(range(self.step)):
            a1, a2 = self.actions[t]
            if a1.zeroQ() and a2.zeroQ():
                continue
            # then at least one of a1 and a2 is non-zero action
            self.real_act[t] = (snap_idx, snap_reward)

            simulated_trace = None
            if not a1.zeroQ():
                simulated_trace = simulator.next_action_some(self.states[t-1], (Action(0), a2))
                simulated_reward = simulated_trace[-1].reward()
                self.zero_1[t] = (len(self.side_chain), simulated_reward)  # (len(self.side_chain), simulated_reward)
                self.side_chain.append(simulated_trace)
            if not a2.zeroQ():
                simulated_trace = simulator.next_action_some(self.states[t-1], (a1, Action(0)))
                simulated_reward = simulated_trace[-1].reward()
                self.zero_2[t] = (len(self.side_chain), simulated_reward)
                self.side_chain.append(simulated_trace)
            if not a1.zeroQ() and not a2.zeroQ():
                # prepare simulation with (a1, a2) = (0, 0), following time t-1
                snap_trace = simulator.next_empty_some(self.states[t - 1])
                self.side_chain.append(snap_trace)
                snap_idx = len(self.side_chain) - 1
            else:
                # the (a1, a2) = (0, 0) trace has been simulated
                snap_trace = simulated_trace
                snap_idx = len(self.side_chain) - 1
            snap_reward = snap_trace[-1].reward()

    def show(self, w=None, nn=None):
        textify = lambda s: s.visual(w, nn)
        for k, s in enumerate(self.states):
            str_s = textify(s)
            if k == 0:
                print('%03d: %s\t\tzero_L\t\tzero_R\t\treal_act' % (k, str_s))
            elif k == len(self.states) - 1:
                print('%03d: %s' % (k, str_s))
            else:
                info1 = self.zero_1[k] if self.zero_1[k] is not None else '  --  '
                info2 = self.zero_2[k] if self.zero_2[k] is not None else '  --  '
                info3 = self.real_act[k] if self.real_act[k] is not None else '  --  '
                print('%03d: %s\t\t%s  \t%s  \t%s' % (k, str_s, info1, info2, info3))
        print(' --- Side Chain Info ---')
        for k, chain in enumerate(self.side_chain):
            r = chain[-1].reward()
            if len(chain) > 1:
                print('[%03d] [%s] => ... => [%s] => [%s] R=%d' % (k, textify(chain[0]), textify(chain[-2]), textify(chain[-1]), r))
            else:
                print('[%03d] [%s] => ... => [%s] R=%d' % (k, textify(chain[0]), textify(chain[-1]), r))

    def split(self, view=0) -> List[Tuple[List[State], List[State]]]:
        """ split pair od advance from trace
        view: 0 - all, 1 - player1 only, 2 - player2 only
        :return List[Tuple[np.ndarray, np.ndarray]] means List[(good_trace, bad_trace)]
        """
        ret = []
        t_hat = None
        for t in reversed(range(self.step)):
            if self.real_act[t] is not None:
                idx_real, reward_real = self.real_act[t]
                # get trace_real
                if idx_real == -1:
                    # [t, end)
                    trace_real = self.states[t:]  # small problem: may encounter loop stable states in tail
                else:
                    trace_real = self.states[t:t_hat] + self.side_chain[idx_real]
                if self.zero_1[t] is not None and view in (0, 1):
                    idx_1, reward_1 = self.zero_1[t]
                    if reward_real > reward_1:
                        trace_1 = self.side_chain[idx_1]
                        ret.append((trace_real, trace_1))
                if self.zero_2[t] is not None and view in (0, 2):
                    idx_2, reward_2 = self.zero_2[t]
                    if reward_real < reward_2:
                        trace_2 = self.side_chain[idx_2]
                        ret.append((trace_real, trace_2))
                t_hat = t  # record branching point as t_hat
        return ret
