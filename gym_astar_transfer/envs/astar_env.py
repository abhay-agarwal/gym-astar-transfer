import os, subprocess, time, signal
import numpy as np
from heapq import *
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

import logging
logger = logging.getLogger(__name__)

Grid = {
    "BLOCK": -1,
    "FREE": 0,
    "PLAYER": 1,
    "END": 2,
    "TRAIL": 3,
    "PATH": 4
}

Directions = {
    0: (0,1),
    1: (0,-1),
    2: (1,0),
    3: (-1,0),
}

space = 0.8 # amount of free space on board
completion_reward = 1 # reward for completion

def astar(grid, start, end):

    start = tuple(start)
    end = tuple(end)
    neighbors = list(Directions.values())
    def h(a,b):
        return (b[0]-a[0]) ** 2 + (b[1]-a[1]) ** 2

    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:h(start,end)}
    oheap = []

    heappush(oheap, (fscore[start], start))
    
    while oheap:

        current = heappop(oheap)[1]

        if current == end:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)          
            tentative_g_score = gscore[current] + h(current,neighbor)
            if neighbor[0] < 0 or neighbor[0] >= grid.shape[0]:
                continue
            if neighbor[1] < 0 or neighbor[1] >= grid.shape[1]:
                continue
            if grid[neighbor[0]][neighbor[1]] == Grid["BLOCK"]:
                continue
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + h(end,neighbor)
                heappush(oheap, (fscore[neighbor], neighbor))
                
    return False

class AStarEnv(gym.Env):

    def __init__(self):

        # Create a random h(x) * w(y) grid of -1 and 0s
        # randomly place a start (1) and an end (2) on the board
        # Use A* to check if a solution exists, otherwise replace start and end

        self.w = 10
        self.h = 10
        self.states = self.w * self.h

        # start with 0-1 vals in all spaces, ration of free space to blocks
        values, probs = [Grid["FREE"], Grid["BLOCK"]], [space, 1-space]
        self.grid = np.random.choice(values, size=(self.h,self.w), p=probs)

        self.action_space = spaces.Discrete(4)
        s = spaces.Discrete(self.states)
        g = spaces.Box(low=-1, high=2, shape=(self.h,self.w))
        self.observation_space = spaces.Tuple((s,g))

        while (True):
            self.start = np.array([np.random.randint(self.h),np.random.randint(self.w)])
            self.end = np.array([np.random.randint(self.h),np.random.randint(self.w)])
            self.path = astar(self.grid, self.start, self.end)
            self.player = self.start

            if self.path and len(self.path) > 10:
                self.grid[self.start[0]][self.start[1]] = Grid["PLAYER"]
                self.grid[self.end[0]][self.end[1]] = Grid["END"]
                self.disp = np.copy(self.grid)
                return

    def _step(self, action):
        self._take_action(action)
        reward = self._get_reward()
        ob = (self._get_state(), self.grid)
        episode_over = not not reward
        return ob, reward, episode_over, self.path

    def _get_state(self):
        return self.player[0] * 10 + self.player[1]

    def _take_action(self, action):
        direction = Directions[action]
        x = self.player[0]+direction[0]
        y = self.player[1]+direction[1]
        if (0 <= x <= self.h) and (0 <= y <= self.w) and self.grid[x][y] != Grid["BLOCK"]:
            self.grid[self.player[0]][self.player[1]] = Grid["FREE"]
            self.disp[self.player[0]][self.player[1]] = Grid["TRAIL"]
            self.grid[x][y] = Grid["PLAYER"]
            self.disp[x][y] = Grid["PLAYER"]
            self.player = np.array([x,y])
            self.path = astar(self.grid, self.player, self.end)

    def _get_reward(self):
        return completion_reward if np.array_equal(self.player,self.end) else 0

    def _reset(self):
        self.grid[self.player[0]][self.player[1]] = Grid["FREE"]
        self.grid[self.start[0]][self.start[1]] = Grid["PLAYER"]
        self.grid[self.end[0]][self.end[1]] = Grid["END"]
        self.player = self.start
        self.path = astar(self.grid, self.player, self.end)

    def _render(self, mode='human', close=False):
        if not close:
            display = np.copy(self.disp)
            for x,y in self.path[1:]:
                display[x,y] = Grid["PATH"]
            print(display)
