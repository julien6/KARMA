import yaml
import os
import numpy as np
import supersuit as ss
import torch
import json
import warnings
import time

from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Discrete, Box
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from marllib import marl
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class cluster_0(ParallelEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, topology_path, model_path, max_steps=200, num_agents=None, max_replica = 5):
        self.topology_path = topology_path
        self.model_path = model_path
        self.max_steps = max_steps
        self.max_replica = max_replica

        # Charger la topologie
        with open(self.topology_path, "r") as f:
            self.topology = json.load(f)

        self.num_services = len(self.topology["services"])
        self.num_metrics = len(self.topology["metrics"])
        self.agents = [f"agent_{agent}" for agent in range(
            0, num_agents or self.num_services)]

        # Espaces d'état et d'action par agent
        self.observation_spaces = {
            f"agent_{i}": spaces.Box(
                low=0,
                high=np.inf,
                shape=(self.num_services, self.num_metrics),
                dtype=np.float32,
            )
            for i in range(self.num_agents)
        }

        self.action_spaces = {
            f"agent_{i}":  spaces.Discrete(self.num_services * self.max_replica)
            for i in range(self.num_agents)
        }

        # Initialiser les agents
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.possible_agents = self.agents.copy()

        # Charger le modèle
        self.model = self._load_model()

        # Autres attributs
        self.steps = 0
        self.states = None

    def _load_model(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(self.num_services *
                            self.num_metrics + self.num_services, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.num_services * self.num_metrics),
        )

        state_dict = torch.load(
            self.model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Modèle chargé depuis {self.model_path}.")
        return model

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.agents = self.possible_agents.copy()
        self.states = np.random.rand(
            self.num_services, self.num_metrics).astype(np.float32)

        # Générer les observations initiales
        observations = {}
        service_idx = 0
        for i, agent in enumerate(self.agents):
            observations[agent] = self.states

        return observations

    def reward_function(self, state, action, next_state):
        """
        Exemple de fonction de récompense pour les tests.
        Calcule une récompense basée sur la distance entre l'état actuel et l'état suivant.
        """
        return -np.sum(np.abs(next_state - state))

    def step(self, actions):
        next_states = np.zeros_like(self.states)
        rewards = {}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        for agent, action in actions.items():
            action_array = np.zeros((self.num_services,))

        for i in range(self.num_services):
            action_array[i] = 
            action_array[service_idx:service_idx +
                        num_services] = actions[agent]
            service_idx += num_services

        # Préparer l'entrée pour le modèle
        input_data = np.concatenate(
            (self.states.flatten(), action_array)).astype(np.float32)
        input_tensor = torch.tensor(input_data).unsqueeze(0)

        # Prédire l'état suivant
        next_state_flat = self.model(input_tensor).squeeze(0).detach().numpy()
        next_states = next_state_flat.reshape(
            self.num_services, self.num_metrics)

        # Calculer les récompenses et générer les nouvelles observations
        observations = {}
        service_idx = 0
        for i, agent in enumerate(self.agents):
            num_services = self.services_per_agent + \
                (1 if i < self.extra_services else 0)
            current_state = self.states[service_idx:service_idx + num_services]
            action = action_array[service_idx:service_idx + num_services]
            next_state = next_states[service_idx:service_idx + num_services]

            rewards[agent] = self.reward_function(
                state=current_state.mean(axis=0),
                action=action.mean(),
                next_state=next_state.mean(axis=0),
            )
            observations[agent] = next_state
            service_idx += num_services

        self.states = next_states
        self.steps += 1

        # Vérifier si l'épisode est terminé
        if self.steps >= self.max_steps:
            terminations = {agent: True for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self, mode="human"):
        if mode == "human":
            print("État actuel :")
            print(self.states)

    def seed(self, seed=None):
        np.random.seed(seed)

    def close(self):
        print("Fermeture de l'environnement.")


# register all scenario with env class
REGISTRY = {}
REGISTRY["cluster_0"] = cluster_0

policy_mapping_dict = {
    "cluster_0": {
        "description": "Simple K8s cluster",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    }
}


class RLlibKarmaCluster(MultiAgentEnv):

    def __init__(self, env_config):
        map = env_config["map_name"]
        env_config.pop("map_name", None)
        env = REGISTRY[map](**env_config)

        # Aplatir l'espace d'observation
        env = ss.flatten_v0(env)

        # keep obs and action dim same across agents
        # pad_action_space_v0 will auto mask the padding actions
        # env = ss.pad_observations_v0(env)
        # env = ss.pad_action_space_v0(env)

        self.max_steps = env_config.get("max_steps", 200)

        self.env = ParallelPettingZooEnv(env)

        # self.action_space = spaces.Discrete(
        #     self.env.action_spaces[self.env.agents[0]].n)
        self.action_space = Box(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            shape=(self.env.action_space.shape[0],),
            dtype=self.env.action_space.dtype)

        self.observation_space = GymDict({"obs": Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            shape=(self.env.observation_space.shape[0],),
            dtype=self.env.observation_space.dtype)})

        print("=> ", self.action_space)
        print("> ", self.observation_space)

        self.agents = self.env.agents
        self.num_agents = len(self.agents)
        env_config["map_name"] = map
        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for i in self.agents:
            obs[i] = {"obs": original_obs[i]}
        return obs

    def step(self, action_dict):
        o, r, d, info = self.env.step(action_dict)
        rewards = {}
        obs = {}
        for key in action_dict.keys():
            rewards[key] = r[key]
            obs[key] = {
                "obs": o[key]
            }
        dones = {"__all__": d["__all__"]}
        return obs, rewards, dones, info

    def close(self):
        self.env.close()

    def render(self, mode=None):
        self.env.render()
        time.sleep(0.05)
        return True

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.max_steps,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info


legal_scenarios = ["cluster_0"]


class RLlibKarmaCluster_FCOOP(RLlibKarmaCluster):

    def __init__(self, env_config):
        if env_config["map_name"] not in legal_scenarios:
            raise ValueError("must in: 1.cluster_0")
        super().__init__(env_config)

    def step(self, action_dict):
        o, r, d, info = self.env.step(action_dict)
        reward = 0
        for key in r.keys():
            reward += r[key]
        rewards = {}
        obs = {}
        for key in action_dict.keys():
            rewards[key] = reward/self.num_agents
            obs[key] = {
                "obs": o[key]
            }
        dones = {"__all__": d["__all__"]}
        return obs, rewards, dones, info


if __name__ == "__main__":

    import numpy as np
    import torch

    # Chemins vers les fichiers nécessaires
    topology_path = "../../utils/install_topology.json"
    model_path = "mlp_model.pth"

    print("Création de l'environnement cluster_0...")
    env = cluster_0(topology_path, model_path, max_steps=10, num_agents=3)

    # Test reset
    print("Test de la méthode reset...")
    observations = env.reset()
    print("Observations initiales :")
    for agent, obs in observations.items():
        print(f"{agent}: {obs.shape} - {obs}")

    # Vérification des dimensions des observations
    for agent in env.agents:
        assert env.observation_space(agent).shape == observations[agent].shape, \
            f"Erreur dans l'espace d'observation de {agent}"

    # Générer des actions aléatoires pour chaque agent
    print("\nGénération d'actions aléatoires...")
    actions = {
        agent: env.action_space(agent).sample() for agent in env.agents
    }
    print("Actions générées :")
    for agent, action in actions.items():
        print(f"{agent}: {action}")

    # Test step
    print("\nExécution d'une étape dans l'environnement...")
    observations, rewards, terminations, truncations, infos = env.step(actions)

    # Afficher les résultats de la méthode step
    print("\nRésultats de la méthode step :")
    print("Observations :")
    for agent, obs in observations.items():
        print(f"{agent}: {obs}")

    print("\nRécompenses :")
    for agent, reward in rewards.items():
        print(f"{agent}: {reward}")

    print("\nTerminations :")
    print(terminations)

    print("\nTruncations :")
    print(truncations)

    print("\nInfos :")
    print(infos)

    # Vérification de la progression des étapes
    assert env.steps == 1, f"Le compteur de pas n'est pas correct. Actuel : {env.steps}, Attendu : 1"

    # Test multi-étapes
    print("\nTest sur plusieurs étapes...")
    for _ in range(3):
        actions = {
            agent: env.action_space(agent).sample() for agent in env.agents
        }
        observations, rewards, terminations, truncations, infos = env.step(
            actions)
        print(f"Étape {env.steps} :")
        print(f"Observations : {observations}")
        print(f"Récompenses : {rewards}")

    # Vérification de la fin de l'épisode
    print("\nTest de la fin de l'épisode...")
    for _ in range(env.max_steps - env.steps):
        actions = {
            agent: env.action_space(agent).sample() for agent in env.agents
        }
        observations, rewards, terminations, truncations, infos = env.step(
            actions)

    assert all(terminations.values(
    )), "L'épisode ne se termine pas correctement après le nombre maximal de pas."

    print("\nTests terminés avec succès pour l'environnement PettingZoo simple.")

    # print("="*30)
    # print("="*30)

    # # YAML de configuration
    # yaml_config = {
    #     "env": "cluster_0",
    #     "env_args": {
    #         "topology_path": topology_path,
    #         "model_path": model_path,
    #         "max_steps": 10,
    #         "num_agents": 3,
    #     },
    #     "mask_flag": False,
    #     "global_state_flag": True,
    #     "opp_action_in_cc": False,
    #     "agent_level_batch_update": True,
    # }
    # yaml_path = "test_env_config.yaml"
    # with open(yaml_path, "w") as f:
    #     yaml.dump(yaml_config, f)

    # print(f"Configuration YAML sauvegardée à {yaml_path}.")

    # # Charger l'environnement MARLlib
    # env = marl.make_env(
    #     environment_name="cluster",
    #     map_name="cluster_0",
    #     abs_path=yaml_path
    # )
    # print("Environnement MARLlib chargé avec succès.")

    # # Tester reset
    # print("\nTest de la méthode reset()...")
    # observations = env.reset()
    # print(f"Observations initiales : {observations}")

    # # Générer des actions aléatoires
    # print("\nGénération d'actions aléatoires...")
    # actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    # print(f"Actions générées : {actions}")

    # # Tester step
    # print("\nExécution de la méthode step()...")
    # observations, rewards, dones, infos = env.step(actions)
    # print(f"Observations : {observations}")
    # print(f"Récompenses : {rewards}")
    # print(f"Dones : {dones}")
    # print(f"Infos : {infos}")

    # # Vérifier la fin de l'épisode
    # print("\nExécution de plusieurs étapes jusqu'à la fin de l'épisode...")
    # while not dones["__all__"]:
    #     actions = {agent: env.action_space(
    #         agent).sample() for agent in env.agents}
    #     observations, rewards, dones, infos = env.step(actions)

    # print("Épisode terminé.")
    # print(f"Observations finales : {observations}")
    # print(f"Récompenses finales : {rewards}")

    # # Supprimer le fichier YAML de test
    # os.remove(yaml_path)
    # print("Fichier YAML de test supprimé.")
