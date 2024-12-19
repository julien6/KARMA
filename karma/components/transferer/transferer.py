import os
import subprocess
import pickle
import requests
from typing import List


class Transferer:
    def __init__(self, namespace: str, policy_dir: str = "policies"):
        """
        Initializes the Transferer.

        Args:
            namespace (str): The Kubernetes namespace where the agents are deployed.
            policy_dir (str): Directory where agent policy files (.pkl) are stored.
        """
        self.namespace = namespace
        self.policy_dir = os.path.abspath(policy_dir)
        os.makedirs(self.policy_dir, exist_ok=True)

    def load_policies(self, agent_names: List[str]) -> dict:
        """
        Loads policies for the specified agents from .pkl files.

        Args:
            agent_names (List[str]): List of agent names to load policies for.

        Returns:
            dict: A dictionary where keys are agent names and values are loaded policies.
        """
        policies = {}
        for agent_name in agent_names:
            policy_path = os.path.join(self.policy_dir, f"{agent_name}_policy.pkl")
            if os.path.exists(policy_path):
                with open(policy_path, "rb") as f:
                    policies[agent_name] = pickle.load(f)
                    print(f"Loaded policy for agent: {agent_name}")
            else:
                print(f"Warning: Policy file not found for agent: {agent_name}")
        return policies

    def deploy_policies(self, agent_names: List[str]):
        """
        Deploys policies by applying Kubernetes ConfigMaps to the cluster.

        Args:
            agent_names (List[str]): List of agent names whose policies are to be deployed.
        """
        for agent_name in agent_names:
            policy_path = os.path.join(self.policy_dir, f"{agent_name}_policy.pkl")
            if os.path.exists(policy_path):
                configmap_name = f"{agent_name}-policy-config"
                print(f"Creating ConfigMap for agent: {agent_name}...")
                try:
                    subprocess.run([
                        "kubectl", "create", "configmap", configmap_name,
                        "--from-file", policy_path,
                        "--namespace", self.namespace
                    ], check=True)
                    print(f"ConfigMap created for agent: {agent_name}")
                except subprocess.CalledProcessError as e:
                    print(f"Error creating ConfigMap for agent: {agent_name}\n{e}")
            else:
                print(f"Policy file not found for agent: {agent_name}")

    def update_agent_pods(self, agent_names: List[str]):
        """
        Updates Kubernetes pods for agents to use the new policies.

        Args:
            agent_names (List[str]): List of agent names to update pods for.
        """
        for agent_name in agent_names:
            deployment_name = f"{agent_name}-deployment"
            print(f"Restarting pods for agent: {agent_name}...")
            try:
                subprocess.run([
                    "kubectl", "rollout", "restart", f"deployment/{deployment_name}",
                    "--namespace", self.namespace
                ], check=True)
                print(f"Pods restarted for agent: {agent_name}")
            except subprocess.CalledProcessError as e:
                print(f"Error restarting pods for agent: {agent_name}\n{e}")

    def execute_actions(self, service_url: str, agent_name: str, actions: List[dict]):
        """
        Sends actions for an agent to its corresponding service endpoint.

        Args:
            service_url (str): The URL of the agent's service in the cluster.
            agent_name (str): Name of the agent performing actions.
            actions (List[dict]): A list of action dictionaries to send.
        """
        print(f"Executing actions for agent: {agent_name}...")
        for action in actions:
            try:
                response = requests.post(f"http://{service_url}/perform_action", json=action)
                if response.status_code == 200:
                    print(f"Action executed successfully: {action}")
                else:
                    print(f"Failed to execute action: {action}. Status code: {response.status_code}")
            except requests.RequestException as e:
                print(f"Error sending action for agent: {agent_name}. Error: {e}")

    def cleanup(self):
        """
        Cleans up old ConfigMaps and other resources related to policies.
        """
        print("Cleaning up old ConfigMaps...")
        try:
            result = subprocess.run([
                "kubectl", "get", "configmaps",
                "--namespace", self.namespace,
                "-o", "name"
            ], capture_output=True, text=True, check=True)
            configmaps = result.stdout.splitlines()
            for configmap in configmaps:
                if "-policy-config" in configmap:
                    print(f"Deleting ConfigMap: {configmap}")
                    subprocess.run(["kubectl", "delete", configmap, "--namespace", self.namespace], check=True)
            print("Cleanup completed.")
        except subprocess.CalledProcessError as e:
            print(f"Error during cleanup: {e}")


# Example usage
if __name__ == "__main__":
    namespace = "karma-cluster"
    transferer = Transferer(namespace)

    # Define agent names
    agent_names = ["agent1", "agent2"]

    # Load policies
    policies = transferer.load_policies(agent_names)

    # Deploy policies to the cluster
    transferer.deploy_policies(agent_names)

    # Update agent pods
    transferer.update_agent_pods(agent_names)

    # Execute actions (example)
    actions = [{"action": "move", "direction": "north"}, {"action": "move", "direction": "east"}]
    transferer.execute_actions(service_url="agent1-service.karma-cluster.svc.cluster.local", agent_name="agent1", actions=actions)

    # Clean up old resources
    transferer.cleanup()
