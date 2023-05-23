import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
#from gymnasium import Env
#attempt to use AECEnv from pettingzoo
import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from gymnasium.spaces.utils import flatdim
from pettingzoo.utils.agent_selector import agent_selector
import time

class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class CellEntity(Enum):
    # entity encodings for grid observations
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3

#player controller is now used
class Player:
    '''
    Each players stores its own reward, score, history and current timestep
    It also stores its own ID which can allow each player to be called individually
    using player.name when no controller is used

    player.name is used in self.agents. pettingzoo implementation steps using player.name
    The player object can be recovered using "self.players[self._index_map[player.name]]"
    To get the player idx, use "self._index_map[player.name]"
    '''
    def __init__(self):
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None
        self.id= ""

    def setup(self, position, level, field_size):
        self.history = []
        self.position = position
        self.level = level
        self.field_size = field_size
        self.score = 0

    def set_controller(self, controller):
        self.controller = controller

    #unique player ID for each player
    def set_id(self, id):
        self.id = id

    def step(self, obs):
        return self.controller._step(obs)

    #return the player controller name or player id
    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return self.id

#define make_env here
def make_env(raw_env):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    #print(raw_env)
    def env():
        print(raw_env)
        env = raw_env()
        '''
        internal_render_mode = render_mode if render_mode != "ansi" else "human"
        # This wrapper is only for environments which print results to the terminal
        if render_mode == "ansi":
            env = wrappers.CaptureStdoutWrapper(env)
        '''
        # this wrapper helps error handling for discrete action spaces
        env = wrappers.AssertOutOfBoundsWrapper(env)
        # Provides a wide vareity of helpful user errors
        # Strongly recommended
        env = wrappers.OrderEnforcingWrapper(env)
        print("here")
        return env
    print(env)
    return env

class ForagingEnv(AECEnv):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {"name": "petting_lbf-v1", 
                "render_modes": ["human"],
                "render_fps": 10,
                }

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        players,
        max_player_level,
        field_size,
        max_food,
        sight,
        max_episode_steps,
        force_coop,
        normalize_reward=True,
        grid_observation=False,
        penalty=0.0,
        render_mode= "human",
        sleep_time= 0.5
    ):
        self.logger = logging.getLogger(__name__)
        self.render_mode = render_mode
        #self.seed() - check to know when to remove or not
        #load players as self.agents
        self.players= [Player() for _ in range(players)]
        for idx, player in enumerate(self.players):
            player.set_id("player_"+str(idx))
        
        #self.agents contains player names 
        self.agents = [player.name for player in self.players]
        self.possible_agents = self.agents[:]
        self._index_map = {
            agent: idx for idx, agent in enumerate(self.agents)
        }

        self._agent_selector = agent_selector(self.agents)

        self.field = np.zeros(field_size, np.int32)

        self.penalty = penalty
        
        self.max_food = max_food
        self._food_spawned = 0.0
        self.max_player_level = max_player_level
        self.sight = sight
        self.force_coop = force_coop
        self._game_over = None

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self._normalize_reward = normalize_reward
        self._grid_observation = grid_observation
        #time to sleep after rendering
        self.sleep_time = sleep_time

        #self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(6)] * len(self.players)))
        #self.observation_space = gym.spaces.Tuple(tuple([self._get_observation_space()] * len(self.players)))
        self.viewer = None
        #self.num_agents = len(self.players)

        #init state dimension
        state_dim= 0
        # set spaces
        self._agent_selector = agent_selector(self.agents)
        
        self.action_spaces = dict()
        self.observation_spaces = dict()
        for player in self.players:
            self.action_spaces[player.name] = spaces.Discrete(6)
            self.observation_spaces[player.name] = spaces.Tuple(
                tuple([self._get_observation_space()]))

        '''
        #state dim is fixed for now
        state_dim= len(self._get_observation_space()) * len(self.players)
        self.state_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )
        '''
        self.current_step = 0

        self.current_actions = [None] * self.num_agents

    def observation_space(self, player_name):
        '''
        Takes in player_name and not player ID
        '''
        return self.observation_spaces[player_name]

    def action_space(self, player_name):
        '''
        Takes in player_name and not player ID
        '''
        return self.action_spaces[player_name]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    #must return n_obs as it is called by env.last()
    #must also set truncation, terminations and reward using player.name
    def observe(self, player_name):
        player= self.players[self._index_map[player_name]]
        return self._make_player_obs(player)

    def state(self):
        #the logic here is to get the player item in the self.players list
        #using the index_map which takes the player.name from possible agents
        #to provide a idx value which gives the player object
        states = tuple(
            self._make_player_obs(self.players[
            self._index_map[agent]]).astype(np.float32)\
            for agent in self.possible_agents
        )
        return np.concatenate(states, axis=None)

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """
        if not self._grid_observation:
            field_x = self.field.shape[1]
            field_y = self.field.shape[0]
            # field_size = field_x * field_y

            max_food = self.max_food
            max_food_level = self.max_player_level * len(self.players)

            min_obs = [-1, -1, 0] * max_food + [-1, -1, 0] * len(self.players)
            max_obs = [field_x-1, field_y-1, max_food_level] * max_food + [
                field_x-1,
                field_y-1,
                self.max_player_level,
            ] * len(self.players)
        else:
            # grid observation space
            grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)

            # agents layer: agent levels
            agents_min = np.zeros(grid_shape, dtype=np.float32)
            agents_max = np.ones(grid_shape, dtype=np.float32) * self.max_player_level

            # foods layer: foods level
            max_food_level = self.max_player_level * len(self.players)
            foods_min = np.zeros(grid_shape, dtype=np.float32)
            foods_max = np.ones(grid_shape, dtype=np.float32) * max_food_level

            # access layer: i the cell available
            access_min = np.zeros(grid_shape, dtype=np.float32)
            access_max = np.ones(grid_shape, dtype=np.float32)

            # total layer
            min_obs = np.stack([agents_min, foods_min, access_min])
            max_obs = np.stack([agents_max, foods_max, access_max])

        #new implementation to incoporate pettingzoo
        return gym.spaces.Box(np.float32(min_obs), np.float32(max_obs))

    @classmethod
    def from_obs(cls, obs):
        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, p.level, obs.field.shape)
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(players, None, None, None, None)
        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_food(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_food_location(self, row, col):
        if row > 1 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def spawn_food(self, max_food, max_level):
        food_count = 0
        attempts = 0
        min_level = max_level if self.force_coop else 1

        while food_count < max_food and attempts < 1000:
            attempts += 1
            row = np.random.randint(1, self.rows - 1)
            col = np.random.randint(1, self.cols - 1)

            # check if it has neighbors:
            if (
                self.neighborhood(row, col).sum() > 0
                or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                or not self._is_empty_location(row, col)
            ):
                continue

            self.field[row, col] = (
                min_level
                if min_level == max_level
                # ! this is excluding food of level `max_level` but is kept for
                # ! consistency with prior LBF versions
                else np.random.randint(min_level, max_level)
            )
            food_count += 1
        self._food_spawned = self.field.sum()

    def _is_empty_location(self, row, col):
        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def spawn_players(self, max_player_level):
        for player in self.players:

            attempts = 0
            player.reward = 0

            while attempts < 1000:
                row = np.random.randint(0, self.rows)
                col = np.random.randint(0, self.cols)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        np.random.randint(1, max_player_level + 1),
                        self.field_size,
                    )
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position[0] > 0
                and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                and self.field[player.position[0], player.position[1] + 1] == 0
            )
        elif action == Action.LOAD:
            return self.adjacent_food(*player.position) > 0

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    #return player observation
    def _make_obs(self, player):
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    ),
                    level=a.level,
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                )
                for a in self.players
                if (
                    min(
                        self._transform_to_neighborhood(
                            player.position, self.sight, a.position
                        )
                    )
                    >= 0
                )
                and max(
                    self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    )
                )
                <= 2 * self.sight
            ],
            # todo also check max?
            field=np.copy(self.neighborhood(*player.position, self.sight)),
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
        )
    
    #this can be used to return agent level observation based features
    def _get_info(self):
        return {}

    #return next observation for a player and set the reward, etc
    def _make_player_obs(self, player):
        #bulid the observation numpy array list
        def make_obs_array(observation):
            #using player name
            obs_space = self.observation_space(player.name)
            print(f'obs_space: {obs_space}')
            obs = np.zeros(flatdim(obs_space), dtype=np.float32)
            print(f'len_obs: {flatdim(obs_space)}')
            # obs[: observation.field.size] = observation.field.flatten()
            # self player is always first
            seen_players = [p for p in observation.players if p.is_self] + [
                p for p in observation.players if not p.is_self
            ]

            for i in range(self.max_food):
                obs[3 * i] = -1
                obs[3 * i + 1] = -1
                obs[3 * i + 2] = 0

            for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
                obs[3 * i] = y
                obs[3 * i + 1] = x
                obs[3 * i + 2] = observation.field[y, x]

            for i in range(len(self.players)):
                obs[self.max_food * 3 + 3 * i] = -1
                obs[self.max_food * 3 + 3 * i + 1] = -1
                obs[self.max_food * 3 + 3 * i + 2] = 0

            for i, p in enumerate(seen_players):
                obs[self.max_food * 3 + 3 * i] = p.position[0]
                obs[self.max_food * 3 + 3 * i + 1] = p.position[1]
                obs[self.max_food * 3 + 3 * i + 2] = p.level

            return obs

        def make_global_grid_arrays():
            """
            Create global arrays for grid observation space
            """
            grid_shape_x, grid_shape_y = self.field_size
            grid_shape_x += 2 * self.sight
            grid_shape_y += 2 * self.sight
            grid_shape = (grid_shape_x, grid_shape_y)

            agents_layer = np.zeros(grid_shape, dtype=np.float32)
            for player in self.players:
                player_x, player_y = player.position
                agents_layer[player_x + self.sight, player_y + self.sight] = player.level
            
            foods_layer = np.zeros(grid_shape, dtype=np.float32)
            foods_layer[self.sight:-self.sight, self.sight:-self.sight] = self.field.copy()

            access_layer = np.ones(grid_shape, dtype=np.float32)
            # out of bounds not accessible
            access_layer[:self.sight, :] = 0.0
            access_layer[-self.sight:, :] = 0.0
            access_layer[:, :self.sight] = 0.0
            access_layer[:, -self.sight:] = 0.0
            # agent locations are not accessible
            for player in self.players:
                player_x, player_y = player.position
                access_layer[player_x + self.sight, player_y + self.sight] = 0.0
            # food locations are not accessible
            foods_x, foods_y = self.field.nonzero()
            for x, y in zip(foods_x, foods_y):
                access_layer[x + self.sight, y + self.sight] = 0.0
            
            return np.stack([agents_layer, foods_layer, access_layer])

        def get_agent_grid_bounds(agent_x, agent_y):
            return agent_x, agent_x + 2 * self.sight + 1, agent_y, agent_y + 2 * self.sight + 1
        
        def get_player_reward(observation):
            for p in observation.players:
                if p.is_self:
                    return p.reward
                
        #pettingzoo implementation
        #id= self._index_map(player.name) #get player index from players list
        raw_obs = self._make_obs(player)
        #print(f'obs: {obs}')
        if self._grid_observation:
            #return the grids
            layers = make_global_grid_arrays()
            agents_bound = get_agent_grid_bounds(*player.position)
            n_obs= tuple(layers[:, agents_bound[0]:agents_bound[1], agents_bound[2]:agents_bound[3]])
            print(f"\n\n\{player.name} in if= {n_obs}\n\n")
        else:
            #turn the raw obs to valid obseravation with array
            n_obs = tuple(make_obs_array(raw_obs))
            print(f"\n\n\{player.name} no if= {n_obs}\n\n")
        self.rewards[player.name]= player.reward
        self.truncations[player.name] = self.game_over
        self.terminations[player.name] = self.game_over   
        # check the space of obs
        assert self._get_observation_space().contains(np.array(n_obs, dtype= np.float32)), \
            f"obs space error: player: {player.name} obs: {np.array(n_obs, dtype= np.float32)}, \
                obs_space: {self._get_observation_space()}"
        
        return n_obs

    #no longer used
    def _make_gym_obs(self):
                
        def get_player_reward(observation):
            for p in observation.players:
                if p.is_self:
                    return p.reward
        
        #pettingzoo implementation
        #init all return dicts
        nobs, nreward, ntrunc, nterm = {}, {}, {}, {}
        for player in self.players:
            obs= self._make_player_obs(player)
            id= self.players.index(player)  #get player index from players list
            obs = self._make_obs(player)
            nobs[player] = get_player_reward(obs)
            ntrunc[player] = obs.game_over
            nterm[player] = obs.game_over
            # ninfo = [{'observation': obs} for obs in observations]
        ninfo = {}
            
        # check the space of obs
        print(f"\n\n\nnobs= {nobs}\n\n")
        print(f"\n\n\nnobs= {nreward}\n\n")
        
        return nobs, nreward, nterm, ntrunc, ninfo
    
    #seed included into the reset function
    def reset(self, seed= None, options= None):
        if seed is not None:
            self.seed(seed=seed)
        
        #included to match pettingzoo requirement
        #self.agents = self.possible_agents[:]
        self.rewards = {player_name: 0.0 for player_name in self.agents}
        self._cumulative_rewards = {player_name: 0.0 for player_name in self.agents}
        self.terminations = {player_name: False for player_name in self.agents}
        self.truncations = {player_name: False for player_name in self.agents}
        self.infos = {player_name: {} for player_name in self.agents}

        self.current_actions = [None] * self.num_agents
        self.field = np.zeros(self.field_size, np.int32)
        self.spawn_players(self.max_player_level)
        player_levels = sorted([player.level for player in self.players])

        self.spawn_food(
            self.max_food, max_level=sum(player_levels[:3])
        )

        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()
        self.agent_selection = self._agent_selector.reset()

        #nobs, _, _,_, ninfo = self._make_gym_obs()
        #info= self._get_gym_info()
        #return a second info dict to capture the info warn of pettingzoo
        #return nobs, ninfo

    def step_env(self):
        #for p in self.players:
        #    p.reward = 0
        actions= []
        for i, agent in enumerate(self.agents):
            a = self.current_actions[i]
            action= Action(a) if Action(a) \
                in self._valid_actions[self.players[i]] else Action.NONE 

            # check if actions are valid
            if action not in self._valid_actions[self.players[i]]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                action = Action.NONE
            #add action to actions list
            actions.append(action)

        loading_players = set()
        print(f'actions: {actions}')
        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_players.add(player)

        # and do movements for non colliding players

        print(f'collisions: {collisions}')
        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than an player will arrive at location
                continue
            v[0].position = k

        # finally process the loadings:
        while loading_players:
            # find adjacent food
            player = loading_players.pop()
            frow, fcol = self.adjacent_food_location(*player.position)
            food = self.field[frow, fcol]

            adj_players = self.adjacent_players(frow, fcol)
            adj_players = [
                p for p in adj_players if p in loading_players or p is player
            ]

            adj_player_level = sum([a.level for a in adj_players])

            loading_players = loading_players - set(adj_players)

            if adj_player_level < food:
                # failed to load
                for a in adj_players:
                    a.reward -= self.penalty
                continue

            # else the food was loaded and each player scores points
            for a in adj_players:
                a.reward = float(a.level * food)
                if self._normalize_reward:
                    a.reward = a.reward / float(
                        adj_player_level * self._food_spawned
                    )  # normalize reward
            # and the food is removed
            self.field[frow, fcol] = 0

        # this can be modified to specify truncation and termination later
        # field.sum gives the scenario when all foods has been eaten
        # current step gives the trunction condition
        '''
        self._game_over = (
            self.field.sum() == 0 or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()
        '''
        for p in self.players:
            p.score += p.reward

    #step implementation
    def step(self, action):
        #check if current agent is dead
        #'''
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            '''
            This still needs some touching
            '''
            #print(f'{self.agent_selection} is dead  action is {action}')
            #self._agent_selector.reinit(self.agents)
            if len(self.agents) != 0:
                self.agent_selection = self._agent_selector.next()
            return
        #'''
        #get current agent
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        #next_idx = (current_idx + 1) % self.num_agents
        #set the next agent as the next agent in selection
        #print(f'Next agent in step: {self.agent_selection}')

        self.current_actions[current_idx] = action
        #print(f'action= {action}')

        #print(f'self._agent_selector.is_last(): {self._agent_selector.is_last()}')
        if self._agent_selector.is_last():
            self.step_env()
            self.current_step += 1
            self._game_over = (
            self.field.sum() == 0 or self._max_episode_steps <= self.current_step
            )
        else:
            self._clear_rewards()

        #set the next agent
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()
        #render the game
        if self.render_mode== "human":
            self.render()
            time.sleep(self.sleep_time)

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
