#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File game_rps.py created on 12:06 2018/1/2 
The game of Rock Paper Scissors

@author: Yichi Xiao
@version: 1.0
"""

from interfaces import *
from tree_search import *
import keyboard  # Using module keyboard


class RPSState(State):
    """RPSState: 7x4 dimensions with 0/1/-1 value
    slot 0-4: 0: [1/0/-1] player1/none/player2
    slot 0-4: 1: [1/0] Rock
    slot 0-4: 2: [1/0] Scissors
    slot 0-4: 3: [1/0] Paper
    """
    def __init__(self, s, flag=0):
        super(RPSState, self).__init__(s)
        self.flag = flag

    @staticmethod
    def get_initial_state():
        return RPSState(np.zeros((7, 4)))

    def feature_for_player_x(self, player) -> np.ndarray:
        """ state code with mask player==1
        + 6+5+4+3 = 18 => {6: [0]->[1-7], ..., 3: [3]->[4-7]}
        + 1 => flag for winner
        """
        if player == 1: # player1's view
            this = np.concatenate([a[1:4] * max(a[0],0) for a in self.s])
            that = np.concatenate([a[1:4] * max(-a[0],0) for a in self.s])
        elif player == 2: # player2's view
            this = np.concatenate([a[1:4] * max(-a[0],0) for a in reversed(self.s)])
            that = np.concatenate([a[1:4] * max(a[0],0) for a in reversed(self.s)])
        else:
            raise RuntimeError('player should be 1 or 2')
        f = lambda a,b,c,x,y,z: a*y+b*z+c*x-b*x-c*y-a*z
        idx = 3*7
        features = np.zeros(3*7+6+5+4+3 + 1) # dim=40
        features[0:idx] = this
        for i in range(4):
            for j in range(i+1, 7):
                features[idx] = f(this[i*3], this[i*3+1], this[i*3+2],
                                  that[j*3], that[j*3+1], that[j*3+2])
                idx += 1
        features[idx] = self.flag if player == 1 else -self.flag  # ! this is really important !
        return features

    def feature_for_player1(self) -> np.ndarray:
        return self.feature_for_player_x(1)

    def feature_for_player2(self) -> np.ndarray:
        return self.feature_for_player_x(2)

    def stableQ(self) -> bool:
        return all(a[0] == 0 for a in self.s)

    def terminateQ(self) -> bool:
        return self.flag != 0  #self.s[5, 0] == 1 or self.s[1, 0] == -1

    def reward(self) -> int:
        return self.flag

    def __str__(self):
        if self.flag == 1:
            return '-|-|W|I|N|-|-'
        elif self.flag == -1:
            return '-|-|L|O|S|-|-'
        a = ['-']*7
        for i, s in enumerate(self.s):
            if s[0] > 0:
                a[i] = 'R' if s[1] == 1 else \
                       'S' if s[2] == 1 else 'P'
            if s[0] < 0:
                a[i] = 'r' if s[1] == 1 else \
                       's' if s[2] == 1 else 'p'
        return '|'.join(a)

    def visual(self, w=None, nn=None):
        s = str(self)
        if w is not None:
            d = np.dot(self.feature_func(), w)
            if nn is None:
                s += '(%+.4f)' % d
            else:
                nv = nn.predict_one(self.feature_func())
                s += '(%+.4f|%+.4f)' % (d, nv)
        return s


class RPSAction(Action):
    """RPSAction: 0 - None, 1 - Rock, 2 - Scissors, 3 - Paper"""
    def __init__(self, a):
        super(RPSAction, self).__init__(a)
        self.a = a

    @staticmethod
    def get_action_spaces():
        #return (RPSAction(0), RPSAction(1), )
        return (RPSAction(0), RPSAction(1), RPSAction(2), RPSAction(3))


class RPSSimulator(BaseSimulator):
    """RPSSimulator: Simulator of Rock Paper Scissors"""
    def __init__(self):
        super(RPSSimulator, self).__init__()

    def initial_state(self):
        return RPSState.get_initial_state()

    def _step_env(self, state, copy=True):
        self.step_cnt += 1  # this is the requirement of super class
        f = lambda a, b, c, x, y, z: a * y + b * z + c * x - b * x - c * y - a * z
        s = state.s # name alias
        flag = 0  # indicator
        if copy:
            s = s.copy()
        # 0. check if win
        if s[5, 0] > 0:
            flag = 1
        elif s[1, 0] < 0:
            flag = -1
        # 1. move right (mask = 1)
        for i in reversed(range(7)):
            if s[i, 0] > 0:
                if i < 6 and s[i+1, 0] < 0:
                    r = f(s[i,1],s[i,2],s[i,3],s[i+1,1],s[i+1,2],s[i+1,3])
                    if r > 0: # win
                        s[i + 1, :] = s[i, :] # move right if win
                    elif r == 0: # tie
                        s[i + 1, :] = np.zeros(4) # remove enemy
                        s[i, :] = np.zeros(4) # remove self
                if i < 6 and s[i+1, 0] >= 0:
                    s[i+1, :] = s[i, :] # just move right
                s[i, :] = np.zeros(4) # clear self, when i=6 it also clears
        # 2. move left (mask = -1)
        for i in range(7):
            if s[i, 0] < 0:
                if i > 0 and s[i-1, 0] > 0:
                    r = f(s[i,1],s[i,2],s[i,3],s[i-1,1],s[i-1,2],s[i-1,3])
                    if r > 0: # win
                        s[i - 1, :] = s[i, :] # move left if win
                    elif r == 0:  # tie
                        s[i - 1, :] = np.zeros(4)  # remove enemy
                        s[i, :] = np.zeros(4)  # remove self
                if i > 0 and s[i-1, 0] <= 0:
                    s[i-1, :] = s[i, :] # just move left
                s[i, :] = np.zeros(4) # clear self, when i=0 it also clears

        if copy:
            return RPSState(s, flag=flag) # wrap s with State
        else:
            state.flag = flag
            return state # return the parameter, since the reference has changed

    def next1(self, state, a1: RPSAction, copy=False):
        s = state.s # name alias
        if copy:
            s = s.copy()
        if not a1.zeroQ():
            s[0, 0] = 1
            s[0, a1.a] = 1 # since RPSAction.a in {1,2,3}
        if copy:
            return RPSState(s)
        else:
            return state

    def next2(self, state, a2: RPSAction, copy=False):
        s = state.s # name alias
        if copy:
            s = s.copy()
        if not a2.zeroQ():
            s[6, 0] = -1
            s[6, a2.a] = 1 # since RPSAction.a in {1,2,3}
        if copy:
            return RPSState(s)
        else:
            return state


class RPSRandomPolicy(Policy):
    """ZeroPolicy: a policy which always returns zero action"""
    def __init__(self, limit=0):
        super(RPSRandomPolicy, self).__init__(RPSAction)
        self.action_space = RPSAction.get_action_spaces()
        self.limit = limit
        self.act_count = 0

    def action(self, state, player=1):
        a = random.choice(self.action_space)
        if self.limit > 0 and self.act_count >= self.limit:
            a = RPSAction(0)
        if not a.zeroQ():
            self.act_count += 1
        return a

    def reset(self):
        self.act_count = 0


class HandRPSPolicy(Policy):
    def __init__(self):
        super(HandRPSPolicy, self).__init__(RPSAction)

    def action(self, state: RPSState, player=1):
        s = state.s
        a = 0
        if player == 1:
            if s[6, 0] < 0:
                a = 1 if s[6, 1] > 0 else 2 if s[6, 2] > 0 else 3
            else:
                a = 0
        elif player == 2:
            if s[0, 0] > 0:
                a = 1 if s[0, 1] > 0 else 2 if s[0, 2] > 0 else 3
            else:
                a = 0
        return RPSAction(a)


class InteractiveRPSPolicy(Policy):
    def __init__(self):
        super(InteractiveRPSPolicy, self).__init__(RPSAction)

    def action(self, state: State, player=1):
        print('==> %s  \t R > S > P' % state)
        while True:  # making a loop
            try:  # used try so that if user pressed other than the given key error will not be shown
                if keyboard.is_pressed('1'):  # if key 'q' is pressed
                    a = 1
                    break  # finishing the loop
                elif keyboard.is_pressed('2'):
                    a = 2
                    break
                elif keyboard.is_pressed('3'):
                    a = 3
                    break
                elif keyboard.is_pressed('0'):
                    a = 0
                    break
            except:
                print('1 - Rock, 2 - Scissors, 3 - Paper ')
                pass  # if user pressed other than the given key the loop will break

        keyboard.stash_state()  # remove the key press state
        return RPSAction(a)


def test():
    game = Game(RPSRandomPolicy(limit=10), RPSRandomPolicy(limit=10), max_step=25)
    simulator = RPSSimulator()
    game.reset()
    trace = game.start(simulator)
    trace.show()
    feas = [s.feature_func() for s in trace.states]
    loose_trace = LoosedTrace(trace, simulator)
    loose_trace.show()
    splitted = loose_trace.split(view=0)
    X, y = IRL(simulator, RPSAction(0)).feed([trace])


def dry_try():
    policy0 = BaseTreeSearch(RPSAction, wf.simulator, wf.NN)
    policy1 = RPSRandomPolicy(limit=3)
    traces = wf._repeat_game_with_policy(100, policy0, policy1)
    LoosedTrace(traces[1], wf.simulator).show(wf.IRL.coef, wf.NN)

if __name__ == '__main__':
    from coupling import Workflow
    tp = Exploration(HandRPSPolicy(), epsilon=0.1)
    itp = InteractiveRPSPolicy()
    wf = Workflow(RPSSimulator(), RPSState, RPSAction, test_policy=itp)
    # wf.flow(RPSRandomPolicy(limit=3), RPSRandomPolicy(limit=3))
    wf.command('(define traces0 (repeat 100 random_policy random_policy))')
    wf.command('(train! traces0)')    # train will effect NN => tree_search_policy, not pure module
    wf.command('(define policy_v01 tree_search_policy)')
    wf.command('(repeat 100 policy_v01 random_policy)')
    wf.command('(define traces1 (repeat  50 policy_v01 (explore policy_v01 0.3)))')
    wf.command('(train! traces1)')
    wf.command('(define policy_v02 tree_search_policy)')
    wf.command('(define traces2 (repeat 100 policy_v02 (explore policy_v02 0.3)))')

    # enable this line will start the game play against human input
    # wf.command('(repeat 1 policy_v02 test_policy)')

