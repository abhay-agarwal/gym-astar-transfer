import os, subprocess, time, signal
import numpy as np
import cv2
from heapq import *
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

import logging
logger = logging.getLogger(__name__)

Grid = {
    "BLOCK": 255,
    "PLAYER": 170,
    "END": 85,
    "FREE": 0,
}

Directions = {
    0: (7,0),
    1: (5,5),
    2: (0,7),
    3: (-5,5),
    4: (-7,0),
    5: (-5,-5),
    6: (0,-7),
    7: (5,-5),
}

Inverted = {v: k for k, v in Directions.items()}

default_reward = 1 # reward for matching A*
object_size = 5
finish_dist = 7
max_timesteps = 9

class ThetaStarEnv(gym.Env):

    metadata = {"render.modes": ["human"]}
    
    def __init__(self):

        # [x] Create or load an 'environment' (picture) in greyscale
        # place two grey markers at start and end point (different grey, changes every session
        # Reward is 1/(steps^2)?
        # make start and end further away from each other as you get more training?
        # plot steps-to-distance and see if it goes down

        filename = os.path.join(os.path.dirname(__file__), "env.png")
        self.map = cv2.imread(filename, 0) # greyscale 8bit
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=255, shape=(42,42,1))

        self.max_dist = 12
        self.min_dist = finish_dist

        # while (True):
        #     self.start = np.random.randint(self.map.shape[0]-object_size, size=2)
        #     self.player = self.start
        #     self.end = np.random.randint(self.map.shape[0]-object_size, size=2)
        #     if np.linalg.norm(self.end-self.start) <= self.max_dist:
        #         if self.valid_position(self.start) and self.valid_position(self.end):
        #             break

    def _get_state(self):
        frame = np.copy(self.map)
        py,px = tuple(self.player)
        ey,ex = tuple(self.end)
        cv2.rectangle(frame, (px, py), (px+object_size, py+object_size), Grid["PLAYER"], cv2.FILLED)
        cv2.rectangle(frame, (ex, ey), (ex+object_size, ey+object_size), Grid["END"], cv2.FILLED)
        
        # resize the state to a more manageable 42x42
        frame = cv2.resize(frame, (80, 80))
        frame = cv2.resize(frame, (42, 42))
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        frame = np.reshape(frame, [42, 42, 1])
        return frame

    def valid_position(self, pos):
        # a "position" is the top-left pixel
        x,y = pos
        return not cv2.countNonZero(self.map[x:x+object_size,y:y+object_size])

    def _step(self, action):
        self.steps += 1
        self._take_action(action)
        ob = self._get_state()
        reward = default_reward if np.linalg.norm(self.end-self.player) < finish_dist else 0
        episode_over = reward > 0 or self.steps >= max_timesteps
        return ob, reward, episode_over, {}

    def _take_action(self, action):
        direction = Directions[action]
        x,y = self.player[0]+direction[0], self.player[1]+direction[1]
        h,w = self.map.shape
        if (0 <= x <= h) and (0 <= y <= w) and self.valid_position((x,y)):
            self.player = np.array([x,y])

    def _reset(self):
        self.steps = 0

        while (True):
            self.start = np.random.randint(self.map.shape[0]-object_size, size=2)
            self.player = self.start
            self.end = np.random.randint(self.map.shape[0]-object_size, size=2)
            if self.min_dist < np.linalg.norm(self.end-self.start) <= self.max_dist:
                if self.valid_position(self.start) and self.valid_position(self.end):
                    break

        return self._get_state()

    def increase_difficulty(self, inc=10):
        self.max_dist = min(self.max_dist + inc, 200)
        self.min_dist = max(self.min_dist, self.max_dist - 50)

    def _render(self, mode='human', close=False):
        if not close:
            cv2.imshow("Theta Star", self._get_state())
            cv2.waitKey(5)
        else:
            cv2.destroyAllWindows()
