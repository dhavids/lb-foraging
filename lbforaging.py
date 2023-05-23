import argparse
import logging
import random
import time
import gymnasium as gym
import numpy as np
import lbforaging
from lbforaging.foraging.environment import ForagingEnv

import numpy as np
from pettingzoo.utils.conversions import parallel_wrapper_fn

#initialize ForagingEnv parameters here
#sight means full observation set as size of field f_size
f_size= 8
class raw_env(ForagingEnv):
    '''
    sleep time determines how long each episode timestep will be
    '''
    def __init__(
        self,
        players= 2,
        max_player_level= 2,
        field_size= (f_size,f_size),
        max_food= 3,
        sight= f_size,
        max_episode_steps = 50,
        force_coop= False,
        normalize_reward=True,
        grid_observation=False,
        penalty=0.0,
        render_mode= "human",
        sleep_time= 0.1
    ):
        super().__init__(
            players= players,
            max_player_level= max_player_level,
            field_size= field_size,
            max_food= max_food,
            sight= sight,
            max_episode_steps= max_episode_steps,
            force_coop= force_coop,
            normalize_reward=normalize_reward,
            grid_observation=grid_observation,
            penalty=penalty,
            render_mode= render_mode,
            sleep_time= sleep_time
        )
        self.metadata["name"] = "petting_lbf-v1"

logger = logging.getLogger(__name__)

def _game_loop(env):

    env.reset()

    for agent in env.agent_iter():
        
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample() # this is where you would insert your policy
        
        env.step(action)
        if (reward):
            print(f'Player: {agent} - Reward: {reward}')
    # print(env.players[0].score, env.players[1].score)

def main(game_count=1, render=False):

    env = raw_env()
    #sets the env.render_mode to the argparse input
    env.render_mode= render
    parallel_env = parallel_wrapper_fn(env)

    for episode in range(game_count):
        _game_loop(env)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )

    args = parser.parse_args()
    main(args.times, args.render)
