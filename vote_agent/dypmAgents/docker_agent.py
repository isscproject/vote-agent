# (C) Copyright IBM Corp. 2018
from gym import spaces
from pommerman.runner import DockerAgentRunner
from pommerman import constants
import numpy as np

class DockerAgent(DockerAgentRunner):

    def __init__(self, agent):
        """
        Wraping an agent to make it work with docker

        Parameters
        ----------
        agent : BaseAgent
            an agent
        """
        self._agent = agent

    def act(self, obs, action_space):
        # convert action_space
        action_space = spaces.Discrete(action_space)

        # convert obs
        for key in ["board", "bomb_blast_strength", "bomb_life"]:
            obs[key] = np.array(obs[key], dtype="uint8")
        obs["position"] = tuple(obs["position"])
        obs["teammate"] = constants.Item(obs["teammate"])
        obs["enemies"] = [constants.Item(n) for n in obs["enemies"]]

        return self._agent.act(obs, action_space)

    def init_agent(cls, id, game_type):
        pass

    def shutdown(cls):
        pass

    def episode_end(cls, reward):
        pass
