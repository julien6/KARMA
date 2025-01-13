import json
import subprocess
import time
import numpy as np
import requests
import os
import sqlite3
import torch
import torch.nn as nn
import random

from utils.cluster_util import KarmaCluster, generate_and_deploy_cluster
from pprint import pprint
from gymnasium import spaces
from multiprocessing import Process


class Transferer:

    def __init__(self, prometheus_url, topology_path: str = None, namespace: str = "karma-cluster"):

        self.topology = None
        if topology_path is not None:
            if not os.path.exists(topology_path):
                raise FileNotFoundError(
                    f"Topology file not found: {topology_path}")

            # Charger la topologie depuis le fichier JSON
            with open(topology_path, "r") as f:
                self.topology = json.load(f)

        self.prometheus_url = prometheus_url
        self.namespace = namespace

    def get_state_from_prometheus(self, retries=3, backoff=2):
        """
        Récupère les métriques agrégées par service depuis Prometheus pour définir l'état.

        Args:
            retries (int): Nombre maximum de tentatives.
            backoff (int): Temps d'attente entre chaque tentative (en secondes).

        Returns:
            dict: État agrégé au niveau des services, structuré au format JSON.
        """
        try:
            # Requêtes PromQL pour récupérer les métriques nécessaires
            queries = {
                "cpu_usage": "sum(rate(container_cpu_usage_seconds_total[5m])) by (deployment)",
                "memory_usage": "sum(container_memory_working_set_bytes) by (deployment)",
                "cpu_request": "sum(kube_pod_container_resource_requests{resource='cpu'}) by (deployment)",
                "memory_request": "sum(kube_pod_container_resource_requests{resource='memory'}) by (deployment)",
                "pods_healthy": "count(kube_pod_status_phase{phase='Running'}) by (deployment)",
                "pods_failed": "count(kube_pod_status_phase{phase='Failed'}) by (deployment)",
                "desired_replicas": "sum(kube_deployment_spec_replicas) by (deployment)",
                "current_replicas": "sum(kube_deployment_status_replicas) by (deployment)",
                "network_in": "sum(rate(container_network_receive_bytes_total[5m])) by (deployment)",
                "network_out": "sum(rate(container_network_transmit_bytes_total[5m])) by (deployment)",
                "pending_requests": "sum(http_requests_in_progress{job='mock-service'}) by (deployment)",
                "requests_total": "sum(rate(http_requests_total[5m])) by (deployment)",
                "requests_errors": "sum(rate(http_requests_total{status=~'5.*'}[5m])) by (deployment)"
            }

            state = {}

            for attempt in range(retries):
                try:
                    # Récupérer les métriques pour chaque service
                    for metric, query in queries.items():
                        if self.topology is not None and metric in self.topology["metrics"]:
                            response = requests.get(
                                f"{self.prometheus_url}/api/v1/query",
                                params={"query": query},
                                timeout=5  # Timeout pour chaque requête
                            )
                            response.raise_for_status()
                            results = response.json()["data"]["result"]

                            # Ajouter les résultats au dictionnaire d'état
                            for result in results:
                                deployment = result["metric"].get("deployment")
                                if not deployment:
                                    continue  # Ignorer les résultats sans déploiement
                                value = float(result["value"][1])

                                if self.topology is not None and deployment in list(self.topology["services"].keys()):
                                    if deployment not in state:
                                        state[deployment] = {}
                                    state[deployment][metric] = value

                    # Ajouter des valeurs par défaut pour les métriques manquantes
                    for deployment in state:
                        for metric in queries.keys():
                            if metric not in state[deployment]:
                                state[deployment][metric] = 0.0

                    # Calcul des métriques dérivées
                    for deployment, metrics in state.items():
                        # Éviter la division par 0
                        total_pods = metrics.get("desired_replicas", 1)
                        metrics["pods_healthy_ratio"] = metrics.get(
                            "pods_healthy", 0) / (1 if total_pods == 0 else total_pods)
                        metrics["pods_failed_ratio"] = metrics.get(
                            "pods_failed", 0) / (1 if total_pods == 0 else total_pods)

                        # Éviter la division par 0
                        requests_total = metrics.get("requests_total", 1)
                        metrics["error_rate"] = metrics.get(
                            "requests_errors", 0) / (1 if requests_total == 0 else requests_total)

                        # Éviter la division par 0
                        cpu_request = metrics.get("cpu_request", 1)
                        metrics["replica_utilization_ratio"] = metrics.get(
                            "cpu_usage", 0) / (1 if cpu_request == 0 else cpu_request)

                    return state  # Retourner l'état complet

                except requests.exceptions.RequestException as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < retries - 1:
                        time.sleep(backoff)
                    else:
                        raise  # Relève l'exception après le nombre maximal de tentatives

        except Exception as e:
            print(f"Error retrieving metrics from Prometheus: {e}")
            return {}

    def scale_deployment(self, deployment_name, replicas, timeout=60, interval=5):
        """
        Met à jour le nombre de réplicas pour un déploiement donné et attend que le changement soit effectif.

        Args:
            deployment_name (str): Nom du déploiement Kubernetes.
            replicas (int): Nouveau nombre de réplicas.
            timeout (int): Temps maximum d'attente en secondes pour que le scaling prenne effet (par défaut : 60).
            interval (int): Intervalle de vérification en secondes (par défaut : 5).

        Returns:
            bool: True si le scaling a réussi et que le nombre attendu de réplicas est atteint, False sinon.
        """
        try:
            # Exécuter la commande kubectl pour mettre à jour les réplicas
            response = subprocess.run(
                [
                    "kubectl", "scale", "deployment", deployment_name,
                    f"--replicas={replicas}", "-n", self.namespace
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True  # Lever une exception si la commande échoue
            )
            print(
                f"Scaling deployment '{deployment_name}' to {replicas} replicas...")

            # Attente active pour vérifier le scaling
            elapsed_time = 0
            while elapsed_time < timeout:
                try:
                    # Obtenir l'état actuel des réplicas
                    state = self.get_state_from_prometheus()
                    current_replicas = state.get(
                        "replicas", {}).get(deployment_name, None)

                    if current_replicas == replicas:
                        print(
                            f"Deployment '{deployment_name}' successfully scaled to {replicas} replicas.")
                        return True

                except Exception as e:
                    print(
                        f"Error checking replicas for '{deployment_name}': {e}")

                time.sleep(interval)
                elapsed_time += interval

            print(
                f"Timeout waiting for deployment '{deployment_name}' to scale to {replicas} replicas.")
            return False

        except subprocess.CalledProcessError as e:
            print(
                f"Failed to scale deployment '{deployment_name}': {e.stderr}")
            return False

    def json_to_gym_state(self, raw_state):
        """
        Convertit un état brut JSON en un état compatible GymSpace.
        Args:
            raw_state (dict): État brut récupéré depuis Prometheus.
        Returns:
            gym_state (np.ndarray): État converti au format GymSpace.
        """
        try:
            nb_services = len(self.topology["services"])
            nb_metrics = len(self.topology["metrics"])
            gym_state = np.zeros((nb_services, nb_metrics), dtype=np.float32)

            # Créer une correspondance entre les noms des services et les indices
            service_indices = {service: idx for idx, service in enumerate(
                self.topology["services"].keys())}

            for service, metrics in raw_state.items():
                if service in service_indices:
                    service_idx = service_indices[service]
                    for metric_idx, metric_name in enumerate(self.topology["metrics"]):
                        gym_state[service_idx, metric_idx] = metrics.get(
                            metric_name, 0.0)

            return gym_state
        except Exception as e:
            print(f"Error in json_to_gym_state: {e}")
            return np.zeros((len(self.topology["services"]), len(self.topology["metrics"])), dtype=np.float32)

    def gym_state_to_json(self, gym_state):
        """
        Convertit un état GymSpace en un format JSON lisible.
        Args:
            gym_state (np.ndarray): État au format GymSpace.
        Returns:
            json_state (dict): État reconstruit au format JSON.
        """
        try:
            services = list(self.topology["services"].keys())
            metrics = self.topology["metrics"]
            json_state = {}

            for service_idx, service_name in enumerate(services):
                json_state[service_name] = {
                    metric_name: gym_state[service_idx, metric_idx]
                    for metric_idx, metric_name in enumerate(metrics)
                }

            return json_state
        except Exception as e:
            print(f"Error in gym_state_to_json: {e}")
            return {}

    def gym_action_to_json(self, gym_action):
        """
        Convertit une action GymSpace en un format JSON lisible.

        Args:
            gym_action (np.ndarray): Action au format GymSpace.

        Returns:
            json_action (dict): Action convertie au format JSON.
        """
        try:
            deployment_names = list(self.topology["services"].keys())

            # Associer chaque action au déploiement correspondant
            json_action = {
                deployment: int(pods_change)
                for deployment, pods_change in zip(deployment_names, gym_action)
            }

            return json_action
        except Exception as e:
            print(f"Error in gym_action_to_json: {e}")
            return {}

    def json_to_gym_action(self, json_action):
        """
        Convertit une action JSON en une action compatible GymSpace.

        Args:
            json_action (dict): Action définie au format JSON.

        Returns:
            gym_action (np.ndarray): Action convertie au format GymSpace (Box).
        """
        try:
            deployment_names = list(self.topology["services"].keys())
            nb_services = len(deployment_names)

            # Initialiser le vecteur GymSpace avec des zéros
            gym_action = np.zeros((nb_services,), dtype=np.int32)

            # Créer une correspondance entre les indices et les services
            service_indices = {service: idx for idx,
                               service in enumerate(deployment_names)}

            # Remplir le vecteur GymSpace avec les valeurs correspondantes
            for service, pods_change in json_action.items():
                if service in service_indices:
                    gym_action[service_indices[service]] = pods_change

            return gym_action
        except Exception as e:
            print(f"Error in json_to_gym_action: {e}")
            return np.zeros((len(self.topology['services']),), dtype=np.int32)

    def create_database(self, db_path):
        """
        Crée une base de données SQLite pour stocker les transitions (state, action, next_state).

        Args:
            db_path (str): Chemin vers la base de données SQLite.
        """
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state TEXT,
                action TEXT,
                next_state TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()
        connection.close()
        self.save_database(db_path)
        print(f"Database created and saved at {db_path}")

    def save_database(self, db_path):
        """
        Sauvegarde la base de données SQLite sur le disque.

        Args:
            db_path (str): Chemin vers la base de données SQLite.
        """
        backup_path = f"{db_path}.backup"
        if os.path.exists(db_path):
            with open(db_path, 'rb') as src, open(backup_path, 'wb') as dst:
                dst.write(src.read())
            print(f"Database backed up to {backup_path}")
        else:
            print(f"Database file not found at {db_path}, backup skipped.")

    def execute_policy_loop(self, db_path, policy_path, interval=5):
        """
        Exécute une boucle infinie pour collecter des transitions (s, a, s') en utilisant une politique donnée.
        """
        policy = torch.load(policy_path, map_location=torch.device('cpu'))
        policy.eval()

        deployment_names = list(self.topology["services"].keys())

        while True:
            try:
                # 1. Collecter l'état courant
                raw_state = self.get_state_from_prometheus()
                gym_state = self.json_to_gym_state(raw_state)

                # 2. Aplatir l'état et obtenir l'action depuis la politique
                gym_state_tensor = torch.tensor(
                    gym_state.flatten(), dtype=torch.float32
                ).unsqueeze(0)
                gym_action_tensor = policy(
                    gym_state_tensor
                ).squeeze(0).detach().numpy()
                gym_action = gym_action_tensor.astype(int)
                gym_action = np.array([random.randint(0,3) for i in range(0, len(deployment_names))])
                print(f"chosen action: {gym_action}")

                # 3. Appliquer l'action
                json_action = self.gym_action_to_json(gym_action)
                for deployment, replicas in json_action.items():
                    self.scale_deployment(deployment, replicas)

                # 4. Attendre l'application
                time.sleep(interval)

                # 5. Collecter l'état suivant
                next_raw_state = self.get_state_from_prometheus()
                next_gym_state = self.json_to_gym_state(next_raw_state)

                # 6. Enregistrer la transition dans la base de données
                connection = sqlite3.connect(db_path)
                cursor = connection.cursor()
                cursor.execute("""
                    INSERT INTO transitions (state, action, next_state)
                    VALUES (?, ?, ?)
                """, (json.dumps(gym_state.tolist()), json.dumps(gym_action.tolist()), json.dumps(next_gym_state.tolist())))
                connection.commit()
                connection.close()

                # Sauvegarder la base de données
                self.save_database(db_path)

                print(
                    f"Transition saved: (state={gym_state}, action={gym_action}, next_state={next_gym_state})")

            except Exception as e:
                print(f"Error in policy loop: {e}")
                time.sleep(interval)

    def initialize_and_execute_policy(self, db_path, policy_path, interval=5):
        """
        Vérifie si la base de données existe, la crée si nécessaire,
        puis lance la boucle d'exécution de la politique dans un processus indépendant.

        Args:
            db_path (str): Chemin vers la base de données SQLite.
            policy_path (str): Chemin vers le fichier .pth contenant la politique.
            interval (int): Temps entre chaque itération en secondes.
        """
        if not os.path.exists(db_path):
            print("Database not found. Creating database...")
            self.create_database(db_path)

        # Lancer la boucle de politique dans un processus indépendant
        process = Process(target=self.execute_policy_loop, args=(
            db_path, policy_path, interval))
        process.start()
        print(f"Policy process started with PID {process.pid}.")
        return process

    def create_and_save_policy(self, save_dir="policies", policy_name="policy.pth", hidden_dim=128):
        """
        Crée une politique simple en utilisant la topologie et l'enregistre au format .pth.
        """
        # Déterminer les dimensions des états et des actions
        state_dim = len(self.topology["services"]) * \
            len(self.topology["metrics"])
        action_dim = len(self.topology["services"])

        # Créer le modèle de politique
        policy = SimplePolicy(state_dim, action_dim, hidden_dim)

        # Créer le répertoire de sauvegarde s'il n'existe pas
        os.makedirs(save_dir, exist_ok=True)

        # Chemin complet du fichier
        policy_path = os.path.join(save_dir, policy_name)

        # Enregistrer la politique
        torch.save(policy, policy_path)
        print(f"Policy saved at {policy_path}")

        return policy_path


class SimplePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(SimplePolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Pour obtenir des actions dans une plage relative
        )
        self.init_weights()  # Initialiser les poids

    def init_weights(self):
        """
        Initialise les poids des couches linéaires avec une distribution uniforme.
        """
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                # Xavier Uniform Initialization
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)  # Initialiser les biais à zéro

    def forward(self, state):
        return self.model(state)


if __name__ == '__main__':

    print("=" * 20)
    print("transferer_test.py")
    print("=" * 20)

    # Set if not already done
    # kc = KarmaCluster("../../utils/install_topology.json")
    # kc = generate_and_deploy_cluster("../../utils/install_topology.json")
    # time.sleep(5)

    tc = Transferer(prometheus_url="http://localhost:9090",
                    namespace="karma-cluster", topology_path="../../utils/install_topology.json")

    try:

        # pprint(tc.get_state_from_prometheus())

        # tc.scale_deployment("a", 1)
        # s1 = tc.get_state_from_prometheus()
        # json.dump(s1, open("s1.json", "w+"), indent=4)

        # tc.scale_deployment("a", 3)
        # s2 = tc.get_state_from_prometheus()
        # json.dump(s2, open("s2.json", "w+"), indent=4)

        # s1 = json.load(open("s1.json", "r"))
        # s2 = json.load(open("s2.json", "r"))
        # print(s2, "\n")
        # print(tc.gym_state_to_json(tc.json_to_gym_state(s2)), "\n\n")
        # print(s2 == tc.gym_state_to_json(tc.json_to_gym_state(s2)))

        # # Exemple de définition du Gym Space adapté
        # print("\n\n\n")
        # nb_services = len(tc.topology["services"])
        # nb_metrics = len(tc.topology["metrics"])
        # sp = spaces.Box(low=0, high=np.inf, shape=(
        #     nb_services, nb_metrics), dtype=np.float32, seed=42)
        # print(sp.sample())

        # # Exemple d'action JSON
        # json_action = {
        #     "a": 2,   # Ajouter 2 pods au service A
        #     "b": -1,  # Retirer 1 pod du service B
        #     "c": 0,   # Aucun changement pour le service C
        #     "d": 3    # Ajouter 3 pods au service D
        # }

        # # Conversion JSON → Gym
        # gym_action = tc.json_to_gym_action(json_action)
        # print("\nGym Action:")
        # print(gym_action)

        # # Conversion inverse Gym → JSON
        # reconstructed_json_action = tc.gym_action_to_json(gym_action)
        # print("\nReconstructed JSON Action:")
        # pprint(reconstructed_json_action)

        # # Vérification de la cohérence
        # assert json_action == reconstructed_json_action, "Mismatch between original and reconstructed JSON action!"
        # print("\nJSON ↔ Gym Action conversion is consistent.")

        # Chemin de la base de données et de la politique
        db_path = "../modeler/transition.db"  # "transitions.db"
        policy_dir = "policies"
        policy_name = "policy.pth"
        policy_path = None

        # Créer et sauvegarder une politique
        policy_path = tc.create_and_save_policy(
            save_dir=policy_dir, policy_name=policy_name)

        process = tc.initialize_and_execute_policy(
            db_path, policy_path, interval=5)

        print("Press Ctrl+C to stop the policy process.")
        try:
            process.join()
        except KeyboardInterrupt:
            print("Stopping policy process...")
            process.terminate()
            process.join()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Nettoyage optionnel
        # kc.delete_cluster()
        pass
