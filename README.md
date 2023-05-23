
<p align="center">
 <img width="350px" src="docs/img/logo.png" align="center" alt="Level Based Foraging (LBF)" />
 <p align="center">A multi-agent reinforcement learning environment</p>
</p>

<!-- TABLE OF CONTENTS -->
<h1> Table of Contents </h1>

- [About The Project](#about-the-project)
- [Getting Started](#getting-started)

<!-- ABOUT THE PROJECT -->
# About The Project

This environment is a mixed cooperative-competitive game, which focuses on the coordination of the agents involved. Agents navigate a grid world and collect food by cooperating with other agents if needed.

<p align="center">
 <img width="450px" src="docs/img/lbf.gif" align="center" alt="Level Based Foraging (LBF) illustration" />
</p>

More specifically, agents are placed in the grid world, and each is assigned a level. Food is also randomly scattered, each having a level on its own. Agents can navigate the environment and can attempt to collect food placed next to them. The collection of food is successful only if the sum of the levels of the agents involved in loading is equal to or higher than the level of the food. Finally, agents are awarded points equal to the level of the food they helped collect, divided by their contribution (their level). The figures below show two states of the game, one that requires cooperation, and one more competitive.


While it may appear simple, this is a very challenging environment, requiring the cooperation of multiple agents while being competitive at the same time. In addition, the discount factor also necessitates speed for the maximisation of rewards. Each agent is only awarded points if it participates in the collection of food, and it has to balance between collecting low-levelled food on his own or cooperating in acquiring higher rewards. In situations with three or more agents, highly strategic decisions can be required, involving agents needing to choose with whom to cooperate. Another significant difficulty for RL algorithms is the sparsity of rewards, which causes slower learning.

This is a Python simulator for level based foraging. It is based on OpenAI's RL framework, with modifications for the multi-agent domain. The efficient implementation allows for thousands of simulation steps per second on a single thread, while the rendering capabilities allows humans to visualise agent actions. Our implementation can support different grid sizes or agent/food count. Also, game variants are implemented, such as cooperative mode (agents always need to cooperate) and shared reward (all agents always get the same reward), which is attractive as a credit assignment problem.```


## Action space

actions is a LIST of N INTEGERS (one of each agent) that should be executed in that step. The integers should correspond to the Enum below:

```python
class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5
```
Valid actions can always be sampled like in a gym environment, using:
```python

```
To use the enviroment with pettingzoo, install pettingzoo==1.22.3 and pyglet==1.5 (if you wish to render it). Anohter known dependencies is six
```

Original Project Link: [https://github.com/semitable/lb-foraging](https://github.com/semitable/lb-foraging)

