#!/user/bin/env python
# -*- coding: utf-8 -*-
#
# The interfaces of TreeSearchAgent

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.svm import LinearSVC
from scipy.linalg import norm
import numpy as np
import random
from itertools import repeat, chain
from typing import List, Tuple, Type, Optional


class State(object):
    """State: Abstract class for game state representation"""
    def __init__(self, s):
        self.s = s
        self.cached_nn_score_1 = None
        self.cached_nn_score_2 = None

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

    def score_by(self, nn, view, use_cached_value=True) -> float:
        """
        Cached function of neural network score.
        :param nn: NN neural network
        :param use_cached_value: if True, the value will be cached
        :return: float the value estimated by neural network
        """
        # TODO need some refactor work
        if view == 1:
            if self.cached_nn_score_1 is not None and use_cached_value:
                return self.cached_nn_score_1
            else:
                self.cached_nn_score_1 = nn.predict_one(self.feature_func(view=view))
                return self.cached_nn_score_1
        elif view == 2:
            if self.cached_nn_score_2 is not None and use_cached_value:
                return self.cached_nn_score_2
            else:
                self.cached_nn_score_2 = nn.predict_one(self.feature_func(view=view))
                return self.cached_nn_score_2
        else:
            assert view in (1, 2), 'view should be 1 or 2'
            return 0

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
    def get_action_spaces() -> list:
        raise NotImplementedError

    def zeroQ(self) -> bool:
        return self.a == 0

    def __eq__(self, other):
        return self.a == other.a


class Policy(object):
    """Policy: strategy mapping from State to Action"""
    def __init__(self, action_type: Type[Action]):
        self.action_type = action_type

    def action(self, state: State, player=1) -> Action:
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
        self.step_cnt = 0
        self.act_cnt = 0

    def reset_cnt(self):
        a, b = self.step_cnt, self.act_cnt
        self.step_cnt, self.act_cnt = 0, 0
        return a, b

    def initial_state(self):
        # Note: should return State.get_initial_state()
        raise NotImplementedError

    def _step_env(self, state: State, copy=True):
        # Note: should contains self.step_cnt += 1
        raise NotImplementedError

    def _step_act(self, state: State, a1: Action, a2: Action, copy=False):
        self.act_cnt += 1
        return self.next2(self.next1(state, a1, copy), a2, copy)

    def next1(self, state: State, a1: Action, copy=False) -> State:
        raise NotImplementedError

    def next2(self, state: State, a2: Action, copy=False) -> State:
        raise NotImplementedError

    def next(self, state, a1, a2):
        """ First environment step, then action step """
        s = self._step_env(state, copy=True)
        return self._step_act(s, a1, a2, copy=False)

    def next_empty_some(self, state, f=lambda s: s.stableQ() or s.terminateQ(), max_t=100) -> List[State]:
        s = state
        ss = []
        for i in range(max_t):
            s = self._step_env(s)
            ss.append(s)
            if f(s):
                break
        return ss

    def next_action_some(self, state, actions, f=lambda s: s.stableQ() or s.terminateQ(), max_t=100) -> List[State]:
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
    def __init__(self, policy1: Policy, policy2: Policy, max_step=100):
        self.policy1 = policy1
        self.policy2 = policy2
        self.max_step = max_step
        self.t = 0

    def reset(self):
        self.t = 0
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
        while t < self.max_step and not st.terminateQ():
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
        self.X = []
        self.y = []

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
        self.X += [d * (2*r-1) for d, r in zip(self.data, rs)]
        self.y += rs
        # limit recent 50000 samples
        self.X = self.X[-50000:]
        self.y = self.y[-50000:]
        print('feed data %d - [interfaces.py - Line 258]' % len(self.X))
        return self.X, self.y

    def train_w(self, hyper_param_c=1.0, normalized_norm=1.0):
        # Train with linear svm (with no bias)
        model = LinearSVC(random_state=0, fit_intercept=False)
        model.C = hyper_param_c
        model.fit(self.X, self.y)

        self.coef = np.ravel(model.coef_)
        print('w norm: %.4f - [interfaces.py - Line 268]' % norm(self.coef))
        self.coef /= norm(self.coef) / normalized_norm
        return self.coef

    def _split_advantage(self, trace) -> List[np.ndarray]:
        ret = []
        lt = LoosedTrace(trace, self.bs)
        for winner in (1, 2):  # enumerate the winner in player1 and player2
            split_trace_arr = lt.split(view=winner)
            for good_trace, weak_trace in split_trace_arr:
                good_dis_fea = [pow(self.gamma, k) * s.feature_func(view=winner)
                                * (1.0 / (1 - self.gamma) if k == len(good_trace) - 1 and s.stableQ() else 1)
                                for k, s in enumerate(good_trace)]
                weak_dis_fea = [pow(self.gamma, k) * s.feature_func(view=winner)
                                * (1.0 / (1 - self.gamma) if k == len(weak_trace) - 1 and s.stableQ() else 1)
                                for k, s in enumerate(weak_trace)]
                mu_good = sum(good_dis_fea)
                mu_weak = sum(weak_dis_fea)
                ret.append(mu_good - mu_weak)
                # same flow with view=3-winner
                weak_dis_fea = [pow(self.gamma, k) * s.feature_func(view=3-winner)
                                * (1.0 / (1 - self.gamma) if k == len(good_trace) - 1 and s.stableQ() else 1)
                                for k, s in enumerate(good_trace)]
                good_dis_fea = [pow(self.gamma, k) * s.feature_func(view=3-winner)
                                * (1.0 / (1 - self.gamma) if k == len(weak_trace) - 1 and s.stableQ() else 1)
                                for k, s in enumerate(weak_trace)]
                mu_good = sum(good_dis_fea)
                mu_weak = sum(weak_dis_fea)
                ret.append(mu_good - mu_weak)

        return ret

    def process_trace_to_vector(self, trace: List[State], player: int, ret) -> List[Tuple[State, np.ndarray, float]]:
        # for player #
        features = [s.feature_func(view=player) for s in trace]
        features[-1] *= 1.0 / (1 - self.gamma) if trace[-1].stableQ() else 1
        r = 0
        for st, fea in zip(reversed(trace), reversed(features)):
            r = np.dot(fea, self.coef) + self.gamma * r
            ret.append((st, fea, r))

    def _split_branch(self, trace, w) -> List[Tuple[State, np.ndarray, float]]:
        """
        To collect training data for NN.
        Only the true trace is collected using recursive way.
        Specifically,
            a score (i.e. reward) is related to each state.
        Previous methods:
        The original idea is to collect all the branching point,
        where the state diverges to two directions which have different results.
        Specifically,
            depending on different winner, it works in different manner.
            Taking winner 1 as example, two traces are split depending on
            whether player 1 plays the action or plays the zero action.
        But it is not enough, the states along the way should also be collected.
        Specifically,
            each trace split in the above part has successors, and each of them
            can be a training example.
        Also, the terminal state is VERY important.
            So each ending state of trace branch is used.
        """
        ret = []
        actual_trace = trace.states
        # for player 1
        self.process_trace_to_vector(actual_trace, 1, ret)
        self.process_trace_to_vector(actual_trace, 2, ret)

        if ret is not None:
            return ret

        #if winner > 0 and (actual_trace[-1].terminateQ() or actual_trace[-1].stableQ()):
        #    length = len(actual_trace)
        #    for m in range(1, length - 1):
        #        process(self.gamma, actual_trace[m:], winner, ret)
        if not isinstance(trace, LoosedTrace):
            lt = LoosedTrace(trace, self.bs)
        else:
            lt = trace
        for winner in (1, 2):  # enumerate the winner in player1 and player2
            split_trace_arr = lt.split(view=winner)
            for good_trace, weak_trace in split_trace_arr:
                ret += self.process_trace_to_vector(good_trace, winner)
                ret += self.process_trace_to_vector(weak_trace, winner)

        # process terminal state of each side chain
        for chain in random.sample(lt.side_chain, k=1):
            winner = 1 if chain[-1].reward() > 0 \
                else 2 if chain[-1].reward() < 0 else 0
            if winner > 0:
                ret += self.process_trace_to_vector([chain[-1]], winner)

        # fix at the tail of trace
        # Note: it is moved to coupling.py since lt.fix(...) depends on neural network
        # fix_results = lt.fix(...)

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
        self._ready = False

    def build(self):
        model = Sequential()
        hs = self.hidden_units  # just name alias
        assert len(hs) > 0
        model.add(Dense(hs[0], input_dim=self.input_dim, activation='relu'))
        for n in hs[1:]:
            model.add(Dense(n, activation='relu'))  # hidden layers
        model.add(Dense(self.output_dim))  # output value with no activation
        adam = keras.optimizers.Adam(lr=2e-4)
        model.compile(loss='mean_squared_error', optimizer=adam)
        return model

    def train(self, data, labels):
        self.model.fit(data, labels, validation_split=0.2, epochs=self.epochs,
                       batch_size=self.batch_size)
        self._ready = True

    def predict(self, data):
        return self.model.predict(data)

    def predict_one(self, input_feature: np.ndarray) -> float:
        return self.predict(input_feature.reshape(1, -1))[0, 0]

    def save(self, name='model.h5'):
        self.model.save_weights(name)

    def load(self, name='model.h5'):
        self.model.load_weights(name)
        self._ready = True

    def ready(self):
        """ Whether the network is ready to predict """
        return self._ready



class Trace(object):
    """Trace: Trajectory with shape like (s_0, a_1, s_1, ..., a_n, s_n) """
    def __init__(self, s0: State, state_list=None, action_list=None):
        self.states = [s0]  # type: List[State]
        # default value (placeholder) for action[0]
        self.actions = [(Action(0), Action(0))]  # type: List[Tuple[Action, Action]]
        self.step = 0
        if state_list is not None and action_list is not None:
            assert len(state_list) == len(action_list)
            self.states += state_list
            self.actions += action_list
            self.step += len(state_list)

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
        self.side_chain = []  # type: List[List[State]] # contains simulated trace (but with no actions)
        self._loose(simulator)

    def _loose(self, simulator: BaseSimulator):
        """
        Find some branching point and generate side chains
        More info could be found by calling self.show()
        """
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
                # the (a1, a2) = (0, 0) trace has been simulated, reuse it
                snap_trace = simulated_trace
                snap_idx = len(self.side_chain) - 1
            snap_reward = snap_trace[-1].reward()

    def show(self, w=None, nn=None):
        textify = lambda s: s.visual(w, nn)
        for k, s in enumerate(self.states):
            a1, a2 = self.actions[k]
            str_s = str(a1) + textify(s) + str(a2)
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
        """
        split pair at branching points on advance from trace,
        this will use the results of _loose()
        view: 0 - all, 1 - player1 only, 2 - player2 only
        :return list of pairs of traces, i.e. List[(good_trace, bad_trace)]
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

    def fix_auto(self, simulator: BaseSimulator, nn: NN, target=0, arg:dict=None) -> List[List[State]]:
        """
        An wrapper for fix, auto determine the view as win state do not need to fix
        :param target which agent to consider: 0 - all, 1 - player1 only, 2 - player2 only
        """
        reward = self.final_state().reward()
        if reward > 0 and target in (0, 2):
            # player 1 win, should fix as view = 2 => target contains 2
            return self.fix(simulator, nn, view=2, arg=arg)
        elif reward < 0 and target in (0, 1):
            # player 2 win, should fix as view = 1 => target contains 1
            return self.fix(simulator, nn, view=1, arg=arg)
        else:
            # tie, do not consider this case here.
            return []

    def fix(self, simulator: BaseSimulator, nn: NN, view=1, arg:dict=None) -> List[List[State]]:
        reward = self.final_state().reward()
        if reward > 0:
            return []   # It is meaningless to fix win state, except it has magnitude. (we do not consider it here)
        ret = []
        flag = False
        depth = 16
        at = type(self.actions[-1][0])  # type: Type[Action]
        action_space = at.get_action_spaces()
        assert view in (1, 2), 'view should be 1 or 2'
        for t in reversed(range(max(self.step - depth, 0), self.step)):
            st = self.states[t]
            a1, a2 = self.actions[t + 1]
            if view == 1:
                next_states = [simulator.next(st, a, a2) for a in action_space]
                picked_act = a1
            else:
                next_states = [simulator.next(st, a1, a) for a in action_space]
                picked_act = a2
            picked_act_idx = [k for k, a in enumerate(action_space) if a == picked_act][0]
            scores = np.ravel(nn.predict(np.array([s.feature_func(view) for s in next_states])))
            #if np.max(scores) != scores[picked_act_idx]:
            #    print('Mismatch! scores=%s, act_idx=%d' % (scores, picked_act_idx))
            #    print('State:', st, 'with', st.s)
            # enumerate flip actions
            for flip_act in action_space:
                if flip_act == picked_act:
                    continue
                actions = (flip_act, a2) if view == 1 else (a1, flip_act)
                sim_states = simulator.next_action_some(st, actions)
                sim_reward = sim_states[-1].reward()
                if sim_reward > reward:     # fix operation succeed
                    # use the simulation results (due to its head node)
                    arg[self.step - t] += 1
                    flag = True
                    ret.append(sim_states)

        if flag:
            arg[-1] += 1
        return ret