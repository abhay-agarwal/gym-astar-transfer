import os, subprocess, time, signal
import numpy as np
from heapq import *
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from enum import Enum

import logging
logger = logging.getLogger(__name__)

class Grid(Enum):
    BLOCK = -1
    FREE = 0
    START = 1
    END = 2

space = 0.8

def astar(grid, start, end):

    start = tuple(start)
    end = tuple(end)
    neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
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
            if grid[neighbor[0]][neighbor[1]] == Grid.BLOCK.value:
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
        ## b[w][h] , b[x][y]

        # Create a randomly sized w x h grid of zeros
        # randomly place a start (1) and an end (2) on the board
        # randomly place obstacles on the grid (-1)
        # implement A* to make a path to the end.
        # Use A* to check if a solution exists.

        self.w = 10
        self.h = 10
        self.states = self.w * self.h
        values, probs = [Grid.FREE.value, Grid.BLOCK.value], [space, 1-space]

        # start with 0-1 vals in all spaces, ration of free space to blocks
        self.grid = np.random.choice(values, size=(self.w,self.h), p=probs)

        while (True):
            self.start = np.array([np.random.randint(self.w),np.random.randint(self.h)])
            self.end = np.array([np.random.randint(self.w),np.random.randint(self.h)])
            if (np.linalg.norm(self.end-self.start) > 5 and astar(self.grid, self.start, self.end)):
                self.grid[self.start[0]][self.start[1]] = Grid.START.value
                self.grid[self.end[0]][self.end[1]] = Grid.END.value
                break

        self.action_space = spaces.Discrete(4)
        s = spaces.Discrete(self.states)
        g = spaces.Box(low=-1, high=2, shape=(self.w,self.h))
        self.observation_space = spaces.Tuple((s,g))

#     def __del__(self):
#         self.env.act(hfo_py.QUIT)
#         self.env.step()
#         os.kill(self.server_process.pid, signal.SIGINT)
#         if self.viewer is not None:
#             os.kill(self.viewer.pid, signal.SIGKILL)

#     def _configure_environment(self):
#         """
#         Provides a chance for subclasses to override this method and supply
#         a different server configuration. By default, we initialize one
#         offense agent against no defenders.
#         """
#         self._start_hfo_server()

#     def _start_hfo_server(self, frames_per_trial=500,
#                           untouched_time=100, offense_agents=1,
#                           defense_agents=0, offense_npcs=0,
#                           defense_npcs=0, sync_mode=True, port=6000,
#                           offense_on_ball=0, fullstate=True, seed=-1,
#                           ball_x_min=0.0, ball_x_max=0.2,
#                           verbose=False, log_game=False,
#                           log_dir="log"):
#         """
#         Starts the Half-Field-Offense server.
#         frames_per_trial: Episodes end after this many steps.
#         untouched_time: Episodes end if the ball is untouched for this many steps.
#         offense_agents: Number of user-controlled offensive players.
#         defense_agents: Number of user-controlled defenders.
#         offense_npcs: Number of offensive bots.
#         defense_npcs: Number of defense bots.
#         sync_mode: Disabling sync mode runs server in real time (SLOW!).
#         port: Port to start the server on.
#         offense_on_ball: Player to give the ball to at beginning of episode.
#         fullstate: Enable noise-free perception.
#         seed: Seed the starting positions of the players and ball.
#         ball_x_[min/max]: Initialize the ball this far downfield: [0,1]
#         verbose: Verbose server messages.
#         log_game: Enable game logging. Logs can be used for replay + visualization.
#         log_dir: Directory to place game logs (*.rcg).
#         """
#         self.server_port = port
#         cmd = self.hfo_path + \
#               " --headless --frames-per-trial %i --untouched-time %i --offense-agents %i"\
#               " --defense-agents %i --offense-npcs %i --defense-npcs %i"\
#               " --port %i --offense-on-ball %i --seed %i --ball-x-min %f"\
#               " --ball-x-max %f --log-dir %s"\
#               % (frames_per_trial, untouched_time, offense_agents,
#                  defense_agents, offense_npcs, defense_npcs, port,
#                  offense_on_ball, seed, ball_x_min, ball_x_max,
#                  log_dir)
#         if not sync_mode: cmd += " --no-sync"
#         if fullstate:     cmd += " --fullstate"
#         if verbose:       cmd += " --verbose"
#         if not log_game:  cmd += " --no-logging"
#         print('Starting server with command: %s' % cmd)
#         self.server_process = subprocess.Popen(cmd.split(' '), shell=False)
#         time.sleep(10) # Wait for server to startup before connecting a player

#     def _start_viewer(self):
#         """
#         Starts the SoccerWindow visualizer. Note the viewer may also be
#         used with a *.rcg logfile to replay a game. See details at
#         https://github.com/LARG/HFO/blob/master/doc/manual.pdf.
#         """
#         cmd = hfo_py.get_viewer_path() +\
#               " --connect --port %d" % (self.server_port)
#         self.viewer = subprocess.Popen(cmd.split(' '), shell=False)

#     def _step(self, action):
#         self._take_action(action)
#         self.status = self.env.step()
#         reward = self._get_reward()
#         ob = self.env.getState()
#         episode_over = self.status != hfo_py.IN_GAME
#         return ob, reward, episode_over, {}

#     def _take_action(self, action):
#         """ Converts the action space into an HFO action. """
#         action_type = ACTION_LOOKUP[action[0]]
#         if action_type == hfo_py.DASH:
#             self.env.act(action_type, action[1], action[2])
#         elif action_type == hfo_py.TURN:
#             self.env.act(action_type, action[3])
#         elif action_type == hfo_py.KICK:
#             self.env.act(action_type, action[4], action[5])
#         else:
#             print('Unrecognized action %d' % action_type)
#             self.env.act(hfo_py.NOOP)

#     def _get_reward(self):
#         """ Reward is given for scoring a end. """
#         if self.status == hfo_py.end:
#             return 1
#         else:
#             return 0

#     def _reset(self):
#         # set the position as the start position.
#         # return the observation (position number, matrix of board)
#         while self.status == hfo_py.IN_GAME:
#             self.env.act(hfo_py.NOOP)
#             self.status = self.env.step()
#         while self.status != hfo_py.IN_GAME:
#             self.env.act(hfo_py.NOOP)
#             self.status = self.env.step()
#         return self.env.getState()

#     def _render(self, mode='human', close=False):
#         """ Viewer only supports human mode currently. """
#         if close:
#             if self.viewer is not None:
#                 os.kill(self.viewer.pid, signal.SIGKILL)
#         else:
#             if self.viewer is None:
#                 self._start_viewer()

# ACTION_LOOKUP = {
#     0 : hfo_py.DASH,
#     1 : hfo_py.TURN,
#     2 : hfo_py.KICK,
#     3 : hfo_py.TACKLE, # Used on defense to slide tackle the ball
#     4 : hfo_py.CATCH,  # Used only by endie to catch the ball
# }
