from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Box, Discrete
import numpy as np


class MoiseMarlEnv(MultiAgentEnv):
    def __init__(self, config):
        """
        Initialise l'environnement MOISE+ MARL.

        Args:
            config (dict): Configuration pour l'environnement, incluant rôles et missions.
        """
        self.num_agents = config.get("num_agents", 3)
        self.state_space = Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_space = Discrete(5)

        # Définition des rôles et missions
        self.roles = config.get("roles", {})
        self.missions = config.get("missions", {})
        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]

        # Initialiser l'état
        self.reset()

    def reset(self):
        """
        Réinitialise l'environnement.
        """
        self.state = {agent_id: self.state_space.sample() for agent_id in self.agent_ids}
        return self.state

    def step(self, actions):
        """
        Avance l'environnement d'un pas de temps en appliquant les actions.

        Args:
            actions (dict): Dictionnaire des actions prises par chaque agent.

        Returns:
            dict: Observations pour chaque agent.
            dict: Récompenses pour chaque agent.
            dict: Indicateur si l'épisode est terminé pour chaque agent.
            dict: Infos supplémentaires.
        """
        rewards = {}
        dones = {}
        infos = {}

        # Calculer les récompenses et transitions basées sur rôles et missions
        for agent_id, action in actions.items():
            current_state = self.state[agent_id]
            next_state = current_state + np.random.normal(0, 0.1, size=current_state.shape)
            self.state[agent_id] = next_state

            role = self.roles.get(agent_id, {})
            mission = self.missions.get(agent_id, {})
            rewards[agent_id] = self._calculate_reward(agent_id, action, role, mission)
            dones[agent_id] = False  # Remplace par la logique de fin d'épisode
            infos[agent_id] = {"role": role, "mission": mission}

        # Exemple simple : fin d'épisode si une métrique atteint un seuil
        done = any(np.linalg.norm(state) > 10 for state in self.state.values())
        dones["__all__"] = done

        return self.state, rewards, dones, infos

    def _calculate_reward(self, agent_id, action, role, mission):
        """
        Calcule la récompense d'un agent en fonction de son rôle et de sa mission.

        Args:
            agent_id (str): Identifiant de l'agent.
            action (int): Action prise par l'agent.
            role (dict): Contraintes de rôle.
            mission (dict): Objectifs intermédiaires.

        Returns:
            float: Récompense calculée.
        """
        reward = 0.0

        # Récompenses basées sur le rôle
        if role.get("type") == "bottleneck_detector":
            reward += -abs(action - 2)  # Ex. éviter les actions extrêmes

        # Récompenses basées sur la mission
        if mission.get("objective") == "minimize_pending_requests":
            reward += -np.sum(self.state[agent_id])

        return reward


if __name__ == "__main__":
    # Configuration de l'environnement
    env_config = {
        "num_agents": 3,
        "roles": {
            "agent_0": {"type": "bottleneck_detector"},
            "agent_1": {"type": "ddos_responder"},
            "agent_2": {"type": "failure_manager"},
        },
        "missions": {
            "agent_0": {"objective": "minimize_pending_requests"},
            "agent_1": {"objective": "maintain_availability"},
            "agent_2": {"objective": "reduce_failures"},
        },
    }

    # Enregistrement de l'environnement
    tune.register_env("moise_plus_marl", lambda config: MoiseMarlEnv(config))

    # Configuration RLlib
    config = {
        "env": "moise_plus_marl",
        "env_config": env_config,
        "multiagent": {
            "policies": {
                "bottleneck_policy": (None, Box(-1.0, 1.0, shape=(10,), dtype=np.float32), Discrete(5), {}),
                "ddos_policy": (None, Box(-1.0, 1.0, shape=(10,), dtype=np.float32), Discrete(5), {}),
                "failure_policy": (None, Box(-1.0, 1.0, shape=(10,), dtype=np.float32), Discrete(5), {}),
            },
            "policy_mapping_fn": lambda agent_id: {
                "agent_0": "bottleneck_policy",
                "agent_1": "ddos_policy",
                "agent_2": "failure_policy",
            }[agent_id],
        },
        "framework": "torch",
        "num_workers": 2,
        "train_batch_size": 200,
    }

    # Lancement de l'entraînement
    tune.run(
        "PPO",
        config=config,
        stop={"episodes_total": 1000},
        checkpoint_at_end=True,
    )
