import argparse
import logging
import random
import time
import gymnasium as gym
import numpy as np
import lbforaging
from lbforaging.foraging.environment import ForagingEnv, make_env

import numpy as np
from pettingzoo.utils.conversions import parallel_wrapper_fn

#initialize ForagingEnv parameters here
#sight means full observation
class raw_env(ForagingEnv):
    def __init__(
        self,
        players= 2,
        max_player_level= 2,
        field_size= (8,8),
        max_food= 3,
        sight=  8,
        max_episode_steps = 10,
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


def _game_loop(env, render):
    """
    
    obs, ninfo = env.reset()
    done = False

    if render:
        env.render()
        time.sleep(0.5)

    while not done:

        actions = env.action_space.sample()

        nobs, nreward, nterm, ntrunc, ninfo = env.step(actions)
        if sum(nreward) > 0:
            print(nreward)

        if render:
            env.render()
            time.sleep(0.5)

        done = np.all(nterm)
    # print(env.players[0].score, env.players[1].score)
    """
    env.reset()
    '''
    if render:
        env.render()
        time.sleep(0.5)
    '''
    for agent in env.agent_iter():
        print(f'agent selection: {env.agent_selection}')
        print(f'current agent: {agent}')
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample() # this is where you would insert your policy
        
        env.step(action)
        
        rews= sum([rew for rew in env.rewards.values()])
        if rews > 0:
            print(f'rews: {rews}')
        print(f'env.cul: {env._cumulative_rewards}')
        '''
        if render:
            env.render()
            time.sleep(0.5)
        '''
    # print(env.players[0].score, env.players[1].score)

def main(game_count=1, render=False):
    #env = gym.make("petting_lbf-v1")
    print(raw_env)
    env = raw_env()
    #sets the env.render_mode to the argparse input
    env.render_mode= render
    print(env)
    parallel_env = parallel_wrapper_fn(env)

    for episode in range(game_count):
        _game_loop(env, render)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )

    args = parser.parse_args()
    main(args.times, args.render)
