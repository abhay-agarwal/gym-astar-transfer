import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='AStarTransfer-v0',
    entry_point='gym_astar_transfer.envs:AStarEnv',
    reward_threshold=1.0,
    nondeterministic = True,
)

register(
    id='ThetaStarTransfer-v0',
    entry_point='gym_astar_transfer.envs:ThetaStarEnv',
    reward_threshold=1.0,
    nondeterministic = True,
)
