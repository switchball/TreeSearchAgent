#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File game_pong.py created on 12:56 2018/1/6 

@author: Yichi Xiao
@version: 1.0
"""
from interfaces import *
import pygame


class PongState(State):
    def __init__(self, s, flag=0):
        super(PongState, self).__init__(s)
        self.flag = flag

    @staticmethod
    def get_initial_state():
        d = {
            'ball_x': 0.5, 'ball_y': 0.5,
            'v_x': random.choice([0.03, -0.03]), 'v_y': np.random.randn()*0.01,
            'p1_y': 0.5, 'p2_y': 0.5,
        }
        return PongState(d)

    @staticmethod
    def _feature_one_hot(ball_y, ball_x, v_x, p1_y, p2_y) -> list:
        h_idx = int((ball_y + 0.05) * 10)  # [0 - 10] (-0.05 0.05 0.15 ... 0.95 1.05)
        b_idx = int((ball_x + 0.05) * 10)  # [0 - 10] (-0.05 0.05 0.15 ... 0.95 1.05)
        p_idx = int(p1_y * 5)  # [0 - 4] (0 0.2 0.4 0.6 0.8 1)
        v_idx = 1 if v_x > 0 else 0
        arr = [0] * (11 + 11 + 5 + 2)
        arr[h_idx] = 1
        arr[11 + b_idx] = 1
        arr[11 + 11 + p_idx] = 1
        arr[11 + 11 + 5 + v_idx] = 1
        return arr

    def feature_for_player1(self) -> np.ndarray:
        return PongState._feature(**self.s)

    def feature_for_player2(self) -> np.ndarray:
        d = self.s
        return PongState._feature(1-d['ball_x'], d['ball_y'], -d['v_x'],
                                  d['v_y'], d['p2_y'], d['p1_y'])

    def feature_func(self, view: int = 1) -> np.ndarray:
        """ common: ball_y, v_y
            different: ball_x, v_x, p_self_x (player1's view)
        """
        ball_x = self.s['ball_x']
        ball_y = self.s['ball_y']
        vx = self.s['v_x']
        vy = self.s['v_y']
        p1y = self.s['p1_y']
        p2y = self.s['p2_y']
        common = [vy]
        term = -1 if ball_x < 0 else 1 if ball_x > 1 else 0
        v1 = [ball_y, ball_x, vx, p1y, p2y]
        v2 = [ball_y, 1-ball_x, -vx, p2y, p1y]

        if view == 1:
            one_hot = PongState._feature_one_hot(*v1)
            return np.array(common + v1 + one_hot + [term, abs(p1y - ball_y)])
        else:
            one_hot = PongState._feature_one_hot(*v2)
            return np.array(common + v2 + one_hot + [-term, abs(p2y - ball_y)])

    def stableQ(self):
        return False

    def terminateQ(self):
        return self.flag != 0

    def reward(self):
        return self.flag

    def __str__(self):
        if self.flag == 1:
            return '-|-|W|I|N|-|-'
        elif self.flag == -1:
            return '-|-|L|O|S|-|-'
        a = int((self.s['ball_x'] + 0.05) * 10)
        k = [' '] * 13
        k[1] = k[-2] = '|'
        k[a+1] = 'o'
        if self.s['v_x'] > 0:
            k[a+2] = '>'
        else:
            k[a] = '<'
        return ''.join(k)

    def visual(self, w=None, nn=None):
        s = str(self)
        if w is not None:
            d = np.dot(self.feature_func(), w)
            if nn is None:
                s += '(%+.2f)' % d
            else:
                nv = nn.predict_one(self.feature_func())
                s += '(%+.2f|%+.2f)' % (d, nv)
        return s


class PongAction(Action):
    """PongAction: 0 - None, 1 - Down, -1 - Up"""
    def __init__(self, a):
        super(PongAction, self).__init__(a)
        self.a = a

    @staticmethod
    def get_action_spaces():
        return (PongAction(0), PongAction(1), PongAction(-1))

    def __str__(self):
        return '-' if self.a == 0 else 'v' if self.a == 1 else '^'


class FollowBallPolicy(Policy):
    def __init__(self):
        super(FollowBallPolicy, self).__init__(PongAction)

    def action(self, state: State, player=1):
        b = state.s['ball_y']
        p = state.s['p1_y'] if player == 1 else state.s['p2_y']
        if b > p + 0.01:
            return PongAction(1)
        elif b < p - 0.01:
            return PongAction(-1)
        else:
            return PongAction(0)

class PongSimulator(BaseSimulator):
    """PongSimulator: Simulator of Pong Game"""
    def __init__(self):
        super(PongSimulator, self).__init__()
        self.gui = False
        self.win = 0
        self.lose = 0
        self.helper = None

    def initial_state(self):
        return PongState.get_initial_state()

    def _step_env(self, state: PongState, copy=True):
        self.step_cnt += 1
        TIME_DELTA = 1
        HALF_PH = 0.1  # Half of paddle height
        s = state.s  # name alis
        flag = 0
        if copy:
            s = s.copy()
        s['ball_x'] += s['v_x'] * TIME_DELTA
        s['ball_y'] += s['v_y'] * TIME_DELTA

        if s['ball_y'] < 0:  # Hit the top
            s['ball_y'] = -s['ball_y']
            s['v_y'] = -s['v_y']

        elif s['ball_y'] > 1:  # Hit the bottom
            s['ball_y'] = 2 - s['ball_y']
            s['v_y'] = -s['v_y']

        # ball in left
        if s['ball_x'] < 0:
            if s['p1_y'] - HALF_PH <= s['ball_y'] <= s['p1_y'] + HALF_PH:
                # Hit the left paddle
                offset = (s['ball_y'] - s['p1_y']) / HALF_PH
                U = np.random.randn() * 0.015
                V = (np.random.randn() + offset*0.05) * 0.03
                s['ball_x'] = -s['ball_x']
                s['v_x'] = max(-s['v_x'] + U, 0.03)  # v_x should > 0
                s['v_y'] = s['v_y'] + V
            else:
                flag = -1

        # ball in right
        if s['ball_x'] > 1:
            if s['p2_y'] - HALF_PH <= s['ball_y'] <= s['p2_y'] + HALF_PH:
                # Hit the right paddle
                offset = (s['ball_y'] - s['p2_y']) / HALF_PH
                U = np.random.randn() * 0.015
                V = (np.random.randn() + offset*0) * 0.03
                s['ball_x'] = 2 - s['ball_x']
                s['v_x'] = min(-s['v_x'] + U, -0.03)  # v_x should < 0
                s['v_y'] = s['v_y'] + V
            else:
                flag = 1

        if copy:
            return PongState(s, flag=flag)  # wrap s with State
        else:
            state.flag = flag
            return state  # return the parameter, since the reference has changed

    def _step_act(self, state: PongState, a1: PongAction, a2: PongAction, copy=False):
        self.act_cnt += 1
        new_state = self.next2(self.next1(state, a1, copy), a2, copy)
        if self.gui:
            self.update(new_state)
        return new_state

    def next1(self, state: PongState, a1: PongAction, copy=False):
        s = state.s  # name alis
        if copy:
            s = s.copy()
        s['p1_y'] += a1.a * 0.04
        s['p1_y'] = max(0.1, min(0.9, s['p1_y']))
        if copy:
            return PongState(s)
        else:
            return state

    def next2(self, state: PongState, a2: PongAction, copy=False):
        s = state.s  # name alis
        if copy:
            s = s.copy()
        s['p2_y'] += a2.a * 0.04
        s['p2_y'] = max(0.1, min(0.9, s['p2_y']))
        if copy:
            return PongState(s)
        else:
            return state

    def use_gui(self, status=False):
        self.gui = status
        if status:
            self.gui_init()

    def gui_init(self):
        WIDTH = 600
        HEIGHT = 600

        pygame.init()
        self.fps = pygame.time.Clock()

        self.window = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
        pygame.display.set_caption('Pong GUI')

    def update(self, state):
        GAME_FPS = 3
        WHITE = (255, 255, 255)
        BALL_COLOR = (44, 62, 80)
        PAD_COLOR = (41, 128, 185)
        BACKGROUND_COLOR = (207, 216, 220)
        SCORE_COLOR = (25, 118, 210)

        WIDTH = 600
        HEIGHT = 600
        BALL_RADIUS = 8
        PAD_WIDTH = 8
        PAD_HEIGHT = HEIGHT * 0.2
        HALF_PAD_WIDTH = PAD_WIDTH // 2
        HALF_PAD_HEIGHT = PAD_HEIGHT // 2
        paddle1_x = HALF_PAD_WIDTH - 1
        paddle2_x = WIDTH + 1 - HALF_PAD_WIDTH

        canvas = self.window
        canvas.fill(BACKGROUND_COLOR)
        pygame.draw.line(canvas, WHITE, [WIDTH // 2, 0], [WIDTH // 2, HEIGHT], 1)
        pygame.draw.line(canvas, WHITE, [PAD_WIDTH, 0], [PAD_WIDTH, HEIGHT], 1)
        pygame.draw.line(canvas, WHITE, [WIDTH - PAD_WIDTH, 0], [WIDTH - PAD_WIDTH, HEIGHT], 1)
        ball_pos = (int(state.s['ball_x'] * WIDTH), int(state.s['ball_y'] * HEIGHT))
        paddle1_pos = (paddle1_x, int(state.s['p1_y'] * HEIGHT))
        paddle2_pos = (paddle2_x, int(state.s['p2_y'] * HEIGHT))

        pygame.draw.circle(canvas, BALL_COLOR, ball_pos, BALL_RADIUS, 0)
        pygame.draw.polygon(canvas, PAD_COLOR, [[paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT],
                                                [paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT],
                                                [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT],
                                                [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT]], 0)
        pygame.draw.polygon(canvas, PAD_COLOR, [[paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT],
                                                [paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT],
                                                [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT],
                                                [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT]], 0)
        font1 = pygame.font.SysFont("Comic Sans MS", 25)
        label1 = font1.render("Score %s" % str(self.win)[:5], 1, SCORE_COLOR)
        canvas.blit(label1, (70, 20))
        font2 = pygame.font.SysFont("Comic Sans MS", 25)
        label2 = font2.render("Score %s" % str(self.lose)[:5], 1, SCORE_COLOR)
        canvas.blit(label2, (WIDTH - 50 - 120, 20))

        if self.helper is not None:
            nn_score, sim_score = self.helper.analysis_state(state)
            label3 = font1.render("NN:  %+.2f" % nn_score, 1, SCORE_COLOR)
            canvas.blit(label3, (70, 70))
            label4 = font1.render("Sim: %+.2f" % sim_score, 1, SCORE_COLOR)
            canvas.blit(label4, (70, 120))

            pygame.draw.polygon(canvas,(132,32,65),[[WIDTH // 2, 150],
                                                    [WIDTH // 2, 160],
                                                    [WIDTH // 2 + int(nn_score*10), 160],
                                                    [WIDTH // 2 + int(nn_score*10), 150]], 0)
            pygame.draw.polygon(canvas,(32,132,65),[[WIDTH // 2, 190],
                                                    [WIDTH // 2, 200],
                                                    [WIDTH // 2 + int(sim_score*10), 200],
                                                    [WIDTH // 2 + int(sim_score*10), 190]], 0)


        pygame.display.update()
        pygame.event.pump()
        self.fps.tick(GAME_FPS)

    def update_score(self, w, t, l):
        self.win = w
        self.lose = l


class Helper(object):
    def __init__(self, simulator : BaseSimulator, nn, gamma, w):
        self.simulator = simulator
        self.nn = nn
        self.gamma = gamma
        self.w = w

    def analysis_state(self, state):
        gamma = self.gamma
        # player1's view
        f1 = state.feature_func(view=1)
        nn1 = self.nn.predict_one(f1)

        sim_trace = self.simulator.next_empty_some(state)
        dis1 = [pow(gamma, k) * s.feature_func(view=1)
                * (1.0 / (1 - gamma) if k == len(sim_trace) - 1 else 1)
                for k, s in enumerate(sim_trace)]
        mu = sum(dis1)
        r = np.dot(mu, self.w)

        return nn1, r


if __name__ == '__main__':
    from coupling import Workflow
    simulator = PongSimulator()
    wf = Workflow(simulator, PongState, PongAction, test_policy=FollowBallPolicy())
    if True:
        wf.command('(define traces1 (repeat 1000 random_policy random_policy))')
        wf.command('(train! traces1)')  # train! is not pure
        wf.command('(get_weight)')
        wf.command('(define policy_v01 tree_search_policy)')
        wf.command('(save_model!)')
    else:
        wf.command('(load_model!)')
        #wf.command('(get_weight)')
        wf.command('(define policy_v01 tree_search_policy)')

    #wf.game_simulator.use_gui(True)
    #wf.command('(define k (repeat 100 policy_v01 policy_v01))')
    #wf.command('(draw (car k) (get_weight))')

    #wf.command('(define traces2 (repeat 100 policy_v01 random_policy))')
    #wf.command('(define traces2 (repeat 100 random_policy policy_v01))')
    #t = wf.command('traces2')
    #LoosedTrace(t[2], wf.simulator).show(wf.IRL.coef, wf.NN)

    wf.command('(define policy_v01 tree_search_policy)')
    wf.command('(define traces2 (repeat 200 policy_v01 test_policy))')
    # wf.command('(draw (car traces0) (get_weight))')

    w = wf._train_feature_weight_with_traces(wf.command('traces2'))

    for x in range(2, 7):
        wf.command('(train! traces%s)' % x)
        wf.command('(define policy_v0%s tree_search_policy)' % x)
        wf.command('(define traces%s (repeat 200 policy_v0%s test_policy))'
                   % (x+1,x))


        wf.command('(save_model!)')

    wf.game_simulator.use_gui(True)
    wf.game_simulator.helper = Helper(wf.game_simulator, wf.NN, 0.95, w)
    wf.command('(define t (repeat 100 policy_v01 test_policy))')

