from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import numpy as np

class KubernetesEnv(ParallelEnv):
    def __init__(self, cluster_topology):
        # Initialize the environment
        super().__init__()
        self.cluster_topology = cluster_topology
        self.services = cluster_topology["services"]
        self.connections = cluster_topology["connections"]

        self.agents = list(self.services.keys()) + ["attacker"]
        self.possible_agents = self.agents.copy()

        # Define observation and action spaces
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=np.inf, shape=(len(self.services[list(self.services.keys())[0]].values()) * len(self.services),), dtype=np.float32)
            for agent in self.agents
        }
        
        alpha = 5  # Maximum number of pods to add/remove
        beta = 10  # Maximum input volume multiplier
        gamma = 1.0  # Maximum corruption percentage
        
        self.action_spaces = {
            agent: spaces.MultiDiscrete([len(self.services), 2 * alpha + 1, 2]) for agent in self.services.keys()
        }
        self.action_spaces["attacker"] = spaces.MultiDiscrete([len(self.services), beta + 1, gamma + 1])

    def reset(self, seed=None):
        # Reset the state of the cluster
        self.state = {
            service: {
                "replicas": service_info["min_replicas"],
                "cpu_used": 0,
                "cpu_allocated": service_info["cpu"],
                "ram_used": 0,
                "ram_allocated": service_info["ram"],
                "pods_healthy": service_info["min_replicas"],
                "pods_failed": 0,
                "received_traffic": 0,
                "transmitted_traffic": 0,
                "pending_requests": 0,
                "latency": 0
            }
            for service, service_info in self.services.items()
        }
        self.time_step = 0
        return {agent: self._get_observation(agent) for agent in self.agents}

    def step(self, actions):
        # Apply actions and update cluster state
        rewards = {}
        for agent, action in actions.items():
            if agent == "attacker":
                self._apply_attack(action)
            else:
                self._apply_defender_action(agent, action)

        # Update cluster state
        self._update_cluster_state()

        # Calculate rewards
        for agent in self.agents:
            rewards[agent] = self._calculate_reward(agent)

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        dones = {agent: self.time_step >= 100 for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        self.time_step += 1
        return observations, rewards, dones, infos

    def _get_observation(self, agent):
        # Generate observation for the agent
        obs = []
        for service_name, service in self.state.items():
            obs.extend([
                service["cpu_used"],
                service["cpu_allocated"],
                service["ram_used"],
                service["ram_allocated"],
                service["pods_healthy"],
                service["pods_failed"],
                service["received_traffic"],
                service["transmitted_traffic"],
                service["pending_requests"],
                service["latency"]
            ])
        return np.array(obs, dtype=np.float32)

    def _apply_attack(self, action):
        # Simulate an attack on the cluster
        target_service = list(self.services.keys())[action[0]]
        input_volume_change = action[1] / beta
        input_type_change = action[2] / 100

        self.state[target_service]["received_traffic"] += input_volume_change * 50  # Increase load artificially
        self.state[target_service]["pending_requests"] += int(input_type_change * 10)  # Add corrupted requests

    def _apply_defender_action(self, service, action):
        # Apply defender actions to a service
        target_service = list(self.services.keys())[action[0]]
        pod_change = action[1] - alpha  # Map from [0, 2*alpha] to [-alpha, +alpha]
        kill_service = action[2]

        if kill_service == 1:
            self.state[target_service]["replicas"] = 0  # Kill all pods of the service
            self.state[target_service]["cpu_used"] = 0
            self.state[target_service]["ram_used"] = 0
            self.state[target_service]["pods_healthy"] = 0
            self.state[target_service]["pods_failed"] = 0
        else:
            self.state[target_service]["replicas"] = np.clip(
                self.state[target_service]["replicas"] + pod_change,
                self.services[target_service]["min_replicas"],
                self.services[target_service]["max_replicas"]
            )

    def _update_cluster_state(self):
        # Update cluster state based on current loads and replicas
        for service_name, service_info in self.state.items():
            replicas = service_info["replicas"]
            throughput = replicas * self.services[service_name]["computation_throughput"]
            service_info["cpu_used"] = replicas * 0.8 * self.services[service_name]["cpu"]  # Simulated usage
            service_info["ram_used"] = replicas * 0.7 * self.services[service_name]["ram"]  # Simulated usage
            service_info["health"] = max(0, 1 - service_info["received_traffic"] / throughput)

    def _calculate_reward(self, agent):
        # Calculate reward for the agent
        if agent == "attacker":
            return sum(1.0 - service["health"] for service in self.state.values())
        else:
            return sum(service["health"] for service in self.state.values())

# Example cluster topology
topology = {
    "services": {
        "A": {
            "cpu": 500,
            "ram": 256,
            "min_replicas": 1,
            "max_replicas": 5,
            "computation_throughput": 100
        },
        "B": {
            "cpu": 400,
            "ram": 512,
            "min_replicas": 1,
            "max_replicas": 3,
            "computation_throughput": 80
        },
        "C": {
            "cpu": 600,
            "ram": 512,
            "min_replicas": 1,
            "max_replicas": 4,
            "computation_throughput": 150
        },
        "D": {
            "cpu": 700,
            "ram": 1024,
            "min_replicas": 1,
            "max_replicas": 6,
            "computation_throughput": 300
        }
    },
    "connections": [
        {"source": "INPUT", "destination": "A"},
        {"source": "A", "destination": "B"},
        {"source": "A", "destination": "C"},
        {"source": "B", "destination": "D"},
        {"source": "C", "destination": "D"},
        {"source": "D", "destination": "OUTPUT"}
    ]
}

# Instantiate the environment
env = KubernetesEnv(topology)
