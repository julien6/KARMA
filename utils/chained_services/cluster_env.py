import json.decoder
import numpy as np
import json

from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
from copy import deepcopy
from pprint import pprint

# TODO:
#  - actions space
#     - attackers
#     - defenders
#  - state sapce
#  - apply_defender_action()
#  - apply_attacker_action()
#  - update_cluster_state()
#  - compute_reward()
#  - reset()


class KubernetesEnv(ParallelEnv):

    def __init__(self, cluster_topology, defender_num=3, attacker_num=1, alpha=5, beta=2, gamma=2, max_step=100, seed=42):
        """
        Initializes the K8s cluster.

        Args:
            cluster_topology (dict): The cluster topology containing information about services and connections.
            defender_num (int): The number of defender agents.
            attacker_num (int): The number of attacker agents.
            alpha (int): Maximum number of pods to add/remove (defender).
            beta (int): Absolute maximum input volume (attacker).
            gamma (int): Absolute maximum data corruption (attacker).
        """

        super().__init__()
        self.seed = seed
        self.max_step = max_step
        self.time_step = 0

        self.agents = [f"defender_{i}" for i in range(
            0, defender_num)] + [f"attacker_{i}" for i in range(0, attacker_num)]
        self.possible_agents = self.agents.copy()

        self.services = cluster_topology["services"]
        self.metrics = cluster_topology["metrics"]
        nb_services = len(self.services)

        self.state = self.init_topology_state(cluster_topology)
        self.init_state = deepcopy(self.state)

        # Same observation for all agents (whole K8s state)
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=np.inf, shape=(
                nb_services, len(self.metrics)), dtype=np.float32, seed=self.seed)
            for agent in self.agents
        }

        self.alpha = 5  # Maximum number of pods to add/remove (defender)
        self.beta = 2  # Absolute maximum input volume (attacker)
        self.gamma = 2  # Absolute maximum data corruption (attacker)

        # Action space for defenders
        self.action_spaces = {
            agent: spaces.MultiDiscrete([nb_services, 2 * self.alpha, 2], seed=self.seed) for agent in self.possible_agents if not "attacker" in agent
        }

        # Action space for attackers
        self.action_spaces = {
            agent: spaces.MultiDiscrete([nb_services, 2 * self.beta, 2 * self.gamma], seed=self.seed) for agent in self.possible_agents if not "defender" in agent
        }

    def dict_to_ndarray(self, json_state):
        services = json_state.keys()
        flattened_data = [[] for i in range(0, len(services))]
        for i, service in enumerate(services):
            flattened_data[i] = [json_state[service][metric_name]
                                 for metric_name in self.metrics]
        return np.array(flattened_data, dtype=np.float32)

    def ndarray_to_dict(self, ndarray_state):
        services = list(self.state.keys())
        json_state = {}
        for i, service in enumerate(services):
            json_state[service] = {
                metric_name: ndarray_state[i, j] for j, metric_name in enumerate(self.metrics)
            }
        return json_state

    def init_topology_state(self, install_topology):
        return {
            service_name: {metric_name: service.get(metric_name, 0) for metric_name in self.metrics} for service_name, service in install_topology["services"].items()
        }

    def reset(self, seed=None):
        self.agents = deepcopy(self.possible_agents)
        # Reset the state of the cluster
        self.state = deepcopy(self.init_state)
        # Reset rewards
        self.rewards = {agent: 0 for agent in self.agents}
        # Reset dones
        self.dones = {agent: False for agent in self.agents}
        # Reset infos
        self.infos = {agent: {} for agent in self.agents}
        # Reset time step
        self.time_step = 0
        # Return initial observations
        return {agent: self._get_observation(agent) for agent in self.agents}

    def step(self, actions):
        # Apply actions and update cluster state
        rewards = {}
        for agent, action in actions.items():
            if "attacker" in agent:
                self._apply_attacker_action(agent, action)
            if "defender" in agent:
                self._apply_defender_action(agent, action)

        # Update cluster state
        self._update_cluster_state()

        # Compute and assign global reward
        global_reward = self._compute_global_reward()
        for agent in self.agents:
            if "attacker" in agent:
                rewards[agent] = -global_reward
            if "defender" in agent:
                rewards[agent] = global_reward

        observations = {agent: self._get_observation(
            agent) for agent in self.agents}
        dones = {agent: self.time_step >=
                 self.max_step for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        self.time_step += 1
        return observations, rewards, dones, infos

    def _get_observation(self, agent):
        # Generate observation for the agent
        return self.dict_to_ndarray(self.state)

    def _apply_attacker_action(self, agent, action):
        # Simulate an attack on the cluster
        # TODO: Implement to update following variables:
        # "current_replicas",
        # "desired_replicas",
        # "cpu_usage",
        # "memory_usage",
        # "pods_healthy",
        # "pods_failed",
        # "network_in",
        # "network_out",
        # "pending_requests",
        # "latency",
        # "requests_total",
        # "requests_errors"
        # Parse the attack action: target service, input volume, corruption
        target_service_idx, input_volume, corruption = action
        target_service = list(self.services.keys())[target_service_idx]

        # Apply the attack
        self.state[target_service]["network_in"] += input_volume - self.beta
        self.state[target_service]["requests_errors"] += corruption - self.gamma

    def _apply_defender_action(self, agent, action):
        # Apply defender actions to a service
        # TODO: Implement to update following variables:
        # "current_replicas",
        # "desired_replicas",
        # "cpu_usage",
        # "memory_usage",
        # "pods_healthy",
        # "pods_failed",
        # "network_in",
        # "network_out",
        # "pending_requests",
        # "latency",
        # "requests_total",
        # "requests_errors"

        # Parse the defender action: target service, replica change, action type
        target_service_idx, replica_change = action
        target_service = list(self.services.keys())[target_service_idx]

        # Adjust the replicas
        current_replicas = self.state[target_service]["current_replicas"]
        desired_replicas = current_replicas + \
            (replica_change - self.alpha)  # Centered on 0
        desired_replicas = np.clip(desired_replicas,
                                   self.services[target_service]["min_replicas"],
                                   self.services[target_service]["max_replicas"])

        self.state[target_service]["desired_replicas"] = desired_replicas

    def _update_cluster_state(self):
        """
        Update the state of the cluster based on current actions, replicas, and dependencies.
        Includes updates to flux metrics (network_in, network_out) based on service dependencies and probabilities.
        """
        for connection in self.connections:
            source = connection["source"]
            destination = connection["destination"]
            throughput = connection.get("throughput", -1)

            if source != "INPUT" and source != "OUTPUT":
                source_replicas = self.state[source]["current_replicas"]
                source_throughput = source_replicas * self.services[source]["computation_throughput"]
                actual_throughput = min(throughput, source_throughput) if throughput > 0 else source_throughput

                # Update destination flux
                self.state[destination]["network_in"] += actual_throughput

            if destination != "OUTPUT" and destination != "INPUT":
                destination_replicas = self.state[destination]["current_replicas"]
                destination_capacity = destination_replicas * self.services[destination]["computation_throughput"]
                processed_traffic = min(self.state[destination]["network_in"], destination_capacity)

                # Update outgoing traffic
                self.state[destination]["network_out"] = processed_traffic * 0.9

                # Update remaining network_in after processing
                self.state[destination]["network_in"] -= processed_traffic

        # Update service-specific metrics
        for service, metrics in self.state.items():
            replicas = metrics["current_replicas"]
            metrics["cpu_usage"] = replicas * self.services[service]["cpu_allocated"] * 0.8
            metrics["memory_usage"] = replicas * self.services[service]["ram_allocated"] * 0.7
            metrics["pods_healthy"] = max(0, replicas - metrics["pods_failed"])
            metrics["latency"] = max(0.1, metrics["pending_requests"] / (replicas + 1))

    def _compute_global_reward(self):
        total_health = sum(self.state[service]["pods_healthy"]
                           for service in self.services)
        total_latency = sum(self.state[service]["latency"]
                            for service in self.services)
        total_criticity = sum(self.state[service]["criticity"]
                              for service in self.services)

        # Reward based on health and penalties for latency and criticity
        reward = total_health - total_latency - total_criticity * 10
        return reward

    def render(self, mode="human"):
        '''
        Displays a rendered frame from the environment, if supported.
        Alternate render modes in the default environments are `'rgb_array'`
        which returns a numpy array and is supported by all environments outside
        of classic, and `'ansi'` which returns the strings printed
        (specific to classic environments).
        '''
        raise NotImplementedError

    def close(self):
        '''
        Closes the rendering window.
        '''
        pass

    def seed(self, seed=None):
        '''
         Reseeds the environment (making it deterministic).
         `reset()` must be called after `seed()`, and before `step()`.
        '''
        self.seed = seed

    def state(self):
        '''
        State returns a global view of the environment appropriate for
        centralized training decentralized execution methods like QMIX
        '''
        return self.state


if __name__ == '__main__':

    # Example cluster topology
    topology = json.load(open("./install_topology.json"))

    # Instantiate the environment
    env = KubernetesEnv(topology)

    s0 = env.observation_spaces["defender_0"].sample()

    print(s0)

    unflattened_s0 = env.ndarray_to_dict(s0)

    print("="*20)
    print(unflattened_s0)

    print("="*20)
    flattened_s0 = env.dict_to_ndarray(unflattened_s0)

    print(flattened_s0)
